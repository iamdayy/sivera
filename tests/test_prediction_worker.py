"""Tests for the API-owned PredictionRunner (replaces Celery prediction worker)."""
import asyncio
import io
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import cv2
import numpy as np
import pytest

from api.core.config import Settings
from api.core.storage import LocalStorage
from api.models.orm import Calibration, Camera, PredictionJob
from api.services import prediction_runner_service as service
from api.services.calibration_capture_service import utcnow
from api.services.prediction_orchestrator import prepare_prediction_image
from api.services.prediction_result_service import PredictionArtifacts
from api.services.rtsp_capture_service import CaptureError, CapturedFrame
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
    job = PredictionJob(id=uuid4(), camera_id=uuid4(), status="pending", source_type="upload",
                        source_path="frame.image", source_ts=None)
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
            # Mirror the values the real UPDATE would set.
            values = {col.key: getattr(expr, "value", None) for col, expr in statement._values.items()}
            for name, value in values.items():
                setattr(job, name, value)
            return job.id
        return job
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
    async def fake_vision(*args, **kwargs):
        assert "processing" in runner.commits
        return {"artifacts": {}, "summary": {"volume": None}}
    monkeypatch.setattr(service, "run_vision", fake_vision)
    # Mock pinned_context to return a valid context
    async def fake_context(db, job):
        return {"ids": {"calibration_id": str(job.calibration_id), "ground_model_id": str(job.ground_model_id), "depth_scale_id": str(job.depth_scale_id)}}
    monkeypatch.setattr(service, "pinned_context", fake_context)
    # Mock load_image to return the stored JPEG
    monkeypatch.setattr(runner.worker, "load_image", AsyncMock(return_value=(jpeg(), utcnow(), "frame.image")))
    # Mock artifacts.write and complete
    monkeypatch.setattr(PredictionArtifacts, "write", AsyncMock(return_value=None))
    async def fake_complete(job, result, artifacts, timestamp, path):
        job.status = "done"
        job.result = result["summary"]
        return True
    monkeypatch.setattr(runner.worker, "complete", fake_complete)

    assert asyncio.run(runner.worker.run()) == {"volume": None}
    assert runner.job.status == "done"
    assert asyncio.run(runner.worker.run()) is None


def test_failure_sanitized_and_persisted_in_new_transaction(runner, monkeypatch):
    async def fake_vision(*args, **kwargs):
        raise RuntimeError("rtsp://password@private")
    monkeypatch.setattr(service, "run_vision", fake_vision)
    async def fake_context(db, job):
        return {"ids": {"calibration_id": str(job.calibration_id), "ground_model_id": str(job.ground_model_id), "depth_scale_id": str(job.depth_scale_id)}}
    monkeypatch.setattr(service, "pinned_context", fake_context)
    monkeypatch.setattr(runner.worker, "load_image", AsyncMock(return_value=(jpeg(), utcnow(), "frame.image")))
    monkeypatch.setattr(PredictionArtifacts, "write", AsyncMock(return_value=None))

    assert asyncio.run(runner.worker.run()) is None
    assert runner.job.status == "failed"
    assert runner.job.error_msg == "prediction_processing_failed"
    assert runner.job.completed_at is not None


def test_latest_rechecks_age_when_worker_starts(runner, monkeypatch):
    runner.job.source_type, runner.job.source_hash, runner.job.max_frame_age_sec = "latest_rtsp", "identity", 5
    runner.db.execute.side_effect = None
    runner.db.execute.return_value = SimpleNamespace(one_or_none=lambda: SimpleNamespace(rtsp_url="rtsp://camera/a", stream_enabled=True))
    monkeypatch.setattr(service, "require_rtsp_camera", lambda *args: "identity")
    monkeypatch.setattr(service, "latest_snapshot", AsyncMock(side_effect=CaptureError("snapshot_stale")))
    with pytest.raises(CaptureError, match="snapshot_stale"):
        asyncio.run(runner.worker.load_image(runner.job))
    runner.job.source_hash = "old"
    with pytest.raises(ValueError, match="prediction_source_changed"):
        asyncio.run(runner.worker.load_image(runner.job))


def test_rtsp_frame_uses_fresh_capture_and_actual_timestamp(runner, monkeypatch):
    runner.job.source_type, runner.job.source_hash, runner.job.max_frame_age_sec = "rtsp_frame", "identity", 5
    runner.db.execute.side_effect = None
    runner.db.execute.return_value = SimpleNamespace(one_or_none=lambda: SimpleNamespace(rtsp_url="rtsp://camera/a", stream_enabled=True))
    frame = CapturedFrame(content=jpeg(), width=640, height=480, captured_at=utcnow())
    capture = AsyncMock(return_value=frame)
    monkeypatch.setattr(service, "require_rtsp_camera", lambda *args: "identity")
    monkeypatch.setattr(service, "capture_live_frame", capture)
    image, timestamp, path = asyncio.run(runner.worker.load_image(runner.job))
    assert timestamp == frame.captured_at
    assert image == jpeg()
    assert path is None
    capture.assert_awaited_once()


def test_source_change_during_pipeline_does_not_publish_result(runner, monkeypatch):
    runner.job.source_type, runner.job.source_hash = "rtsp_frame", "old-source"
    monkeypatch.setattr(runner.worker, "load_image", AsyncMock(return_value=(jpeg(), utcnow(), "frame.image")))
    original = runner.db.execute.side_effect
    async def execute(statement):
        if statement.is_update:
            return original(statement)
        return SimpleNamespace(one_or_none=lambda: SimpleNamespace(rtsp_url="rtsp://camera.local/new", stream_enabled=True))
    runner.db.execute.side_effect = execute
    monkeypatch.setattr(service, "require_rtsp_camera", lambda *args: "new-source")
    async def fake_vision(*args, **kwargs):
        return {"artifacts": {}, "summary": {"volume": 123}}
    monkeypatch.setattr(service, "run_vision", fake_vision)
    async def fake_context(db, job):
        return {"ids": {"calibration_id": str(job.calibration_id), "ground_model_id": str(job.ground_model_id), "depth_scale_id": str(job.depth_scale_id)}}
    monkeypatch.setattr(service, "pinned_context", fake_context)
    monkeypatch.setattr(PredictionArtifacts, "write", AsyncMock(return_value=None))

    assert asyncio.run(runner.worker.run()) is None
    assert runner.job.status == "failed" and runner.job.result is None
    assert runner.job.error_msg == "prediction_source_changed"


def test_invalid_intrinsics_rejected_without_remap():
    camera = Camera(resolution_w=640, resolution_h=480)
    calibration = Calibration(k_matrix=np.eye(3).tolist(), k_original=[[float("nan")] * 3] * 3,
                              dist_coeffs=[0] * 5, resolution_w=640, resolution_h=480)
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
