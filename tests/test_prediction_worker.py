import asyncio
import io
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import cv2
import pytest

from api.core.config import Settings
from api.core.storage import LocalStorage
from api.models.orm import Calibration, Camera, PredictionJob
from api.services import prediction_worker_service as service
from api.services.calibration_capture_service import utcnow
from api.services.prediction_orchestrator import prepare_prediction_image
from tests.test_calibration_capture_service import jpeg


def test_explicit_intrinsics_and_legacy_upload_resolution():
    K = [[800, 0, 320], [0, 810, 240], [0, 0, 1]]
    calibration = Calibration(k_matrix=K, dist_coeffs=[0] * 5)
    camera = Camera(resolution_w=640, resolution_h=480)
    image = np.zeros((480, 640, 3), np.uint8)
    assert prepare_prediction_image(image, calibration, camera, "upload") is image
    with pytest.raises(ValueError, match="calibration_intrinsics_missing"):
        prepare_prediction_image(image, calibration, camera, "rtsp_frame")
    calibration.k_original = K
    calibration.k_undist = K
    calibration.resolution_w, calibration.resolution_h = 640, 480
    np.testing.assert_array_equal(prepare_prediction_image(image, calibration, camera, "rtsp_frame"), image)
    with pytest.raises(ValueError, match="prediction_resolution_mismatch"):
        prepare_prediction_image(image[:200], calibration, camera, "latest_rtsp")


@pytest.fixture
def runner(monkeypatch, tmp_path):
    job = PredictionJob(id=uuid4(), camera_id=uuid4(), status="pending", source_type="upload", source_path="frame.image", source_ts=None)
    db = MagicMock(get=AsyncMock(return_value=job), execute=AsyncMock(), no_autoflush=MagicMock(), rollback=AsyncMock())
    commits = []
    @asynccontextmanager
    async def context():
        yield db
        commits.append(job.status)
    async def scalar(statement):
        if statement.is_update:
            if job.status != "pending":
                return None
            job.status = "processing"
            return job.id
        return job.status
    db.scalar = AsyncMock(side_effect=scalar)
    def execute(statement):
        values = {col.key: getattr(expr, "value", None) for col, expr in statement._values.items()}
        for name, value in values.items():
            setattr(job, name, value)
    db.execute.side_effect = execute
    storage = LocalStorage(tmp_path)
    storage.save(job.source_path, io.BytesIO(jpeg()))
    worker = service.PredictionRunner(job.id, context, storage)
    return SimpleNamespace(job=job, db=db, worker=worker, commits=commits)


def test_processing_committed_before_pipeline_and_duplicate_ignored(runner, monkeypatch):
    async def pipeline(db, job, image):
        assert "processing" in runner.commits
        assert image.shape == (480, 640, 3)
        return {"volume": None}
    fake = AsyncMock(side_effect=pipeline)
    monkeypatch.setattr(service, "run_prediction_pipeline", fake)
    assert asyncio.run(runner.worker.run()) == {"volume": None}
    assert runner.job.status == "done"
    assert asyncio.run(runner.worker.run()) is None
    assert fake.await_count == 1


def test_failure_sanitized_and_persisted_in_new_transaction(runner, monkeypatch):
    monkeypatch.setattr(service, "run_prediction_pipeline", AsyncMock(side_effect=RuntimeError("rtsp://password@private")))
    assert asyncio.run(runner.worker.run()) is None
    assert runner.job.status == "failed"
    assert runner.job.error_msg == "prediction_processing_failed"
    assert runner.job.completed_at is not None


def test_latest_rechecks_age_when_worker_starts(runner, monkeypatch):
    runner.job.source_type, runner.job.source_hash, runner.job.max_frame_age_sec = "latest_rtsp", "identity", 5
    runner.db.execute.side_effect = None
    runner.db.execute.return_value = SimpleNamespace(one_or_none=lambda: SimpleNamespace(rtsp_url="rtsp://camera/a", stream_enabled=True))
    monkeypatch.setattr(service, "require_rtsp_camera", lambda *args: "identity")
    monkeypatch.setattr(service, "latest_snapshot", AsyncMock(side_effect=service.CaptureError("snapshot_stale")))
    with pytest.raises(service.CaptureError, match="snapshot_stale"):
        asyncio.run(runner.worker.load_image(runner.job))
    runner.job.source_hash = "old"
    with pytest.raises(ValueError, match="prediction_source_changed"):
        asyncio.run(runner.worker.load_image(runner.job))


def test_rtsp_frame_uses_fresh_capture_and_actual_timestamp(runner, monkeypatch):
    from api.models.orm import CameraSnapshot
    runner.job.source_type, runner.job.source_hash, runner.job.max_frame_age_sec = "rtsp_frame", "identity", 5
    runner.db.execute.side_effect = None
    runner.db.execute.return_value = SimpleNamespace(one_or_none=lambda: SimpleNamespace(rtsp_url="rtsp://camera/a", stream_enabled=True))
    snapshot = CameraSnapshot(storage_key="frame.image", width=640, height=480, captured_at=utcnow())
    capture = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(service, "require_rtsp_camera", lambda *args: "identity")
    monkeypatch.setattr(service, "capture_snapshot", capture)
    image, timestamp, path = asyncio.run(runner.worker.load_image(runner.job))
    assert timestamp == snapshot.captured_at
    assert image.shape == (480, 640, 3)
    capture.assert_awaited_once()


def test_source_change_during_pipeline_does_not_publish_result(runner, monkeypatch):
    runner.job.source_type, runner.job.source_hash = "rtsp_frame", "old-source"
    monkeypatch.setattr(runner.worker, "load_image", AsyncMock(return_value=(np.zeros((480, 640, 3), np.uint8), utcnow(), "frame.image")))
    original = runner.db.execute.side_effect
    async def execute(statement):
        if statement.is_update:
            return original(statement)
        return SimpleNamespace(one_or_none=lambda: SimpleNamespace(rtsp_url="rtsp://camera.local/new", stream_enabled=True))
    runner.db.execute.side_effect = execute
    monkeypatch.setattr(service, "require_rtsp_camera", lambda *args: "new-source")
    monkeypatch.setattr(service, "run_prediction_pipeline", AsyncMock(return_value={"volume": 123}))
    assert asyncio.run(runner.worker.run()) is None
    assert runner.job.status == "failed" and runner.job.result is None
    assert runner.job.error_msg == "prediction_source_changed"


def test_timeout_fence_cannot_be_overwritten_by_completion(runner, monkeypatch):
    async def pipeline(*args):
        runner.job.status = "failed"
        runner.job.error_msg = "prediction_worker_timeout"
        return {"volume": 123}
    monkeypatch.setattr(service, "run_prediction_pipeline", pipeline)
    assert asyncio.run(runner.worker.run()) is None
    assert runner.job.error_msg == "prediction_worker_timeout"
    runner.db.rollback.assert_awaited_once()


def test_invalid_intrinsics_rejected_without_remap():
    camera = Camera(resolution_w=640, resolution_h=480)
    calibration = Calibration(k_matrix=np.eye(3).tolist(), k_original=[[float("nan")]*3]*3,
                              dist_coeffs=[0]*5, resolution_w=640, resolution_h=480)
    with pytest.raises(ValueError, match="calibration_invalid_intrinsics"):
        prepare_prediction_image(np.zeros((480, 640, 3), np.uint8), calibration, camera, "rtsp_frame")


@pytest.mark.parametrize("source_type", ["upload", "rtsp_frame", "latest_rtsp"])
def test_raw_image_undistorts_with_original_not_target_intrinsics(source_type):
    original = np.array([[800., 0, 320], [0, 810, 240], [0, 0, 1]])
    target = np.array([[760., 0, 310], [0, 770, 230], [0, 0, 1]])
    distortion = np.array([.1, -.05, .001, -.001, 0])
    image = np.indices((480, 640)).sum(axis=0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    calibration = Calibration(k_matrix=target.tolist(), k_undist=target.tolist(), k_original=original.tolist(),
                              resolution_w=640, resolution_h=480, dist_coeffs=distortion.tolist())
    camera = Camera(resolution_w=640, resolution_h=480)
    actual = prepare_prediction_image(image, calibration, camera, source_type)
    np.testing.assert_array_equal(actual, cv2.undistort(image, original, distortion, None, target))
    assert not np.array_equal(actual, image)
