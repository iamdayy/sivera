import asyncio
import io
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from PIL import Image

from api.core.config import Settings
from api.core.storage import LocalStorage
from api.models.orm import CalibrationCaptureImage, CalibrationCaptureSession
from api.schemas.calibration_capture import CaptureCreate
from api.services import calibration_capture_service as service
from api.services import rtsp_capture_service as rtsp
from api.workers.celery_app import celery_app
from api.workers.tasks.capture_task import run_capture_task


@pytest.fixture
def anyio_backend():
    return "asyncio"


def session_fixture(**overrides):
    now = datetime.now(timezone.utc)
    values = dict(
        id=uuid4(), camera_id=uuid4(), status="queued", created_at=now,
        deadline_at=now + timedelta(seconds=300), started_at=None, completed_at=None,
        heartbeat_at=None, worker_token=None, last_dispatched_at=None,
        attempted=0, captured=0, failed_attempts=0, interval_sec=1, target_images=2,
        chessboard_cols=9, chessboard_rows=6, square_m=0.025, max_duration_sec=300,
        resolution_w=None, resolution_h=None, error_code=None, finish_reason=None, last_capture_at=None,
        selection_version=0, auto_calibrate=False, accepted=0, rejected=0, rejection_counts={},
        compute_status="none", compute_requested_at=None, compute_dispatched_at=None, compute_started_at=None,
    )
    values.update(overrides)
    return CalibrationCaptureSession(**values)


@pytest.mark.parametrize(("field", "value"), [
    ("interval_sec", 0), ("target_images", 101),
    ("max_duration_sec", 901), ("square_m", float("nan")), ("chessboard_cols", 1),
    ("unknown", True),
])
def test_capture_validation(field, value):
    with pytest.raises(ValueError):
        CaptureCreate(**{field: value})


@pytest.mark.parametrize(("status", "age", "expected", "reason"), [
    ("queued", 121, "failed", "queue_timeout"),
    ("running", 46, "failed", "worker_lost"),
    ("stopping", 0, "stopped", "user_stop"),
    ("completed", 500, "completed", None),
])
def test_recovery_status(status, age, expected, reason):
    now = service.utcnow()
    session = session_fixture(status=status, created_at=now - timedelta(seconds=age), heartbeat_at=now - timedelta(seconds=age))
    service.expire_session(session, now)
    assert session.status == expected
    assert session.finish_reason == reason


def test_deadline_precedes_worker_failure():
    now = service.utcnow()
    session = session_fixture(status="running", deadline_at=now, heartbeat_at=None)
    assert service.expire_session(session, now)
    assert session.status == "expired"
    assert session.error_code is None


def test_relay_uses_reader_credentials_and_uuid_path():
    config = Settings(_env_file=None, MEDIAMTX_ENABLED=True, MEDIAMTX_READ_PASSWORD="p@:/word", MEDIAMTX_HOST="::1")
    camera = uuid4()
    assert rtsp.relay_url(camera, config) == f"rtsp://sivera_reader:p%40%3A%2Fword@[::1]:8554/sivera_{camera}"
    with pytest.raises(rtsp.CaptureError, match="stream_service_not_configured"):
        rtsp.relay_url(camera, Settings(_env_file=None))


def jpeg(width=640, height=480):
    content = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(content, "JPEG")
    return content.getvalue()


def test_frame_dimensions_and_decode_limits():
    frame = rtsp.decode_frame(jpeg())
    assert (frame.width, frame.height) == (640, 480)
    assert frame.captured_at.tzinfo is not None
    with pytest.raises(rtsp.CaptureError, match="capture_resolution_too_large"):
        rtsp.decode_frame(jpeg(), Settings(_env_file=None, CAPTURE_MAX_PIXELS=65536))
    with pytest.raises(rtsp.CaptureError, match="capture_frame_too_large"):
        rtsp.decode_frame(b"x" * 65537, Settings(_env_file=None, CAPTURE_MAX_FRAME_BYTES=65536))
    with pytest.raises(rtsp.CaptureError, match="capture_invalid_frame"):
        rtsp.decode_frame(b"not an image")


@pytest.mark.anyio
async def test_ffmpeg_pipe_bounded_and_no_resize(monkeypatch):
    process = SimpleNamespace(stdout=SimpleNamespace(read=AsyncMock(side_effect=[jpeg(), b""])),
        wait=AsyncMock(return_value=0), returncode=0, pid=12345)
    kill = MagicMock()
    monkeypatch.setattr(rtsp.os, "killpg", kill)
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    config = Settings(_env_file=None, MEDIAMTX_ENABLED=True, MEDIAMTX_READ_PASSWORD="private")
    assert (await rtsp.capture_frame(uuid4(), config)).width == 640
    args = spawn.call_args.args
    assert args[:2] == ("timeout", "--signal=KILL")
    assert "-rtsp_transport" in args and "tcp" in args
    assert args[args.index("-max_pixels") + 1] == str(config.CAPTURE_MAX_PIXELS)
    assert "-vf" not in args and "-s" not in args
    assert spawn.call_args.kwargs["stderr"] == asyncio.subprocess.DEVNULL
    process.returncode = None
    process.stdout.read.side_effect = [b"x" * 65537]
    config.CAPTURE_MAX_FRAME_BYTES = 65536
    with pytest.raises(rtsp.CaptureError, match="capture_frame_too_large"):
        await rtsp.capture_frame(uuid4(), config)
    kill.assert_called_once()


@pytest.mark.anyio
async def test_ffmpeg_timeout_and_cancel_kill_child(monkeypatch):
    async def block(*args):
        await asyncio.Event().wait()
    process = SimpleNamespace(stdout=SimpleNamespace(read=block), wait=AsyncMock(return_value=0), returncode=None, pid=12345)
    kill = MagicMock()
    monkeypatch.setattr(rtsp.os, "killpg", kill)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    config = Settings(_env_file=None, MEDIAMTX_ENABLED=True, MEDIAMTX_READ_PASSWORD="private")
    config.STREAM_CAPTURE_TIMEOUT = 0.01
    with pytest.raises(rtsp.CaptureError, match="capture_timeout"):
        await rtsp.capture_frame(uuid4(), config)
    kill.assert_called_once()
    config.STREAM_CAPTURE_TIMEOUT = 15
    task = asyncio.create_task(rtsp.capture_frame(uuid4(), config))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert kill.call_count == 2


@pytest.fixture
def runner(tmp_path, monkeypatch):
    session = session_fixture()
    camera = SimpleNamespace(rtsp_url="rtsp://camera.local/live", stream_enabled=True)
    db = MagicMock()
    db.get = AsyncMock(return_value=session)
    db.scalar = AsyncMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(one_or_none=lambda: camera))
    images = []
    db.add.side_effect = images.append
    @asynccontextmanager
    async def context():
        yield db
    config = Settings(_env_file=None, MEDIAMTX_ENABLED=True, MEDIAMTX_READ_PASSWORD="private", MEDIAMTX_ALLOWED_RTSP_HOSTS="camera.local")
    worker = service.CaptureRunner(session.id, context, LocalStorage(tmp_path), config)
    async def scalar(statement):
        if statement.is_update:
            if session.status != "queued":
                return None
            session.status = "running"
            session.worker_token = worker.token
            session.heartbeat_at = service.utcnow()
            return session.id
        return session
    db.scalar.side_effect = scalar
    monkeypatch.setattr(service, "capture_frame", AsyncMock(return_value=rtsp.decode_frame(jpeg())))
    monkeypatch.setattr(service, "check_relay_source", AsyncMock())
    async def wait(_):
        return await worker.tick()
    monkeypatch.setattr(worker, "wait", wait)
    return SimpleNamespace(worker=worker, session=session, camera=camera, db=db, images=images)


@pytest.mark.anyio
async def test_runner_completes_with_persisted_progress(runner):
    await runner.worker.run()
    assert runner.session.status == "completed"
    assert runner.session.captured == runner.session.attempted == 2
    assert runner.session.failed_attempts == 0
    assert len(runner.images) == 2
    assert all(isinstance(image, CalibrationCaptureImage) for image in runner.images)
    assert all(runner.worker.storage.exists(image.storage_key) for image in runner.images)
    assert [image.sequence for image in runner.images] == [1, 2]
    assert runner.session.worker_token is None


@pytest.mark.anyio
async def test_runner_bounded_retries(runner, monkeypatch):
    monkeypatch.setattr(service, "capture_frame", AsyncMock(side_effect=rtsp.CaptureError("capture_timeout")))
    await runner.worker.run()
    assert runner.session.status == "failed"
    assert runner.session.attempted == runner.session.failed_attempts == 3
    assert runner.session.captured == 0


@pytest.mark.anyio
async def test_runner_source_disabled(runner):
    runner.camera.stream_enabled = False
    await runner.worker.run()
    assert runner.session.status == "failed"
    assert runner.session.error_code == "capture_stream_disabled"
    assert runner.session.attempted == 0


@pytest.mark.anyio
async def test_runner_source_change_fails_without_new_capture(runner):
    await runner.worker.claim()
    assert await runner.worker.tick()
    runner.camera.rtsp_url = "rtsp://camera.local/new-source"
    assert not await runner.worker.tick(attempt=True)
    assert runner.session.error_code == "capture_source_changed"
    assert runner.session.attempted == 0


@pytest.mark.anyio
async def test_runner_resolution_change_removes_uncommitted_frame(runner, monkeypatch):
    monkeypatch.setattr(service, "capture_frame", AsyncMock(side_effect=[rtsp.decode_frame(jpeg()), rtsp.decode_frame(jpeg(320, 240))]))
    await runner.worker.run()
    assert runner.session.status == "failed"
    assert runner.session.error_code == "capture_resolution_changed"
    assert runner.session.captured == runner.session.failed_attempts == 1
    assert len(list(runner.worker.storage.root.rglob("*.jpg"))) == 1


@pytest.mark.anyio
async def test_stop_cancels_inflight_process(runner, monkeypatch):
    cancelled = asyncio.Event()
    async def slow(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    monkeypatch.setattr(service, "capture_frame", slow)
    task = asyncio.create_task(runner.worker.run())
    await asyncio.sleep(0)
    runner.session.status = "stopping"
    await asyncio.wait_for(task, 2)
    assert cancelled.is_set()
    assert runner.session.status == "stopped"
    assert runner.images == []


@pytest.mark.anyio
async def test_old_worker_fenced_and_duplicate_delivery_ignored(runner):
    await runner.worker.claim()
    runner.session.worker_token = uuid4()
    assert not await runner.worker.tick(attempt=True)
    await runner.worker.run()
    assert runner.images == []


@pytest.mark.anyio
async def test_storage_failure_marks_failed_safely(runner, monkeypatch):
    monkeypatch.setattr(runner.worker.storage, "save", MagicMock(side_effect=RuntimeError("credential private/path")))
    await runner.worker.run()
    assert runner.session.status == "failed"
    assert runner.session.error_code == "capture_worker_error"


@pytest.mark.anyio
async def test_stop_between_storage_write_and_commit_discards_frame(runner, monkeypatch):
    await runner.worker.claim()
    assert await runner.worker.tick(attempt=True)
    save = runner.worker.storage.save
    def stop_after_write(*args):
        result = save(*args)
        runner.session.status = "stopping"
        return result
    monkeypatch.setattr(runner.worker.storage, "save", stop_after_write)
    assert not await runner.worker.save_frame(runner.session.camera_id, rtsp.decode_frame(jpeg()))
    assert runner.session.status == "stopped"
    assert runner.images == []
    assert list(runner.worker.storage.root.rglob("*.jpg")) == []


@pytest.mark.anyio
async def test_source_change_between_storage_write_and_commit_discards_frame(runner, monkeypatch):
    await runner.worker.claim()
    assert await runner.worker.tick(attempt=True)
    save = runner.worker.storage.save
    def change_after_write(*args):
        result = save(*args)
        runner.camera.rtsp_url = "rtsp://camera.local/changed"
        return result
    monkeypatch.setattr(runner.worker.storage, "save", change_after_write)
    assert not await runner.worker.save_frame(runner.session.camera_id, rtsp.decode_frame(jpeg()))
    assert runner.session.error_code == "capture_source_changed"
    assert runner.images == []
    assert list(runner.worker.storage.root.rglob("*.jpg")) == []


@pytest.mark.anyio
async def test_queue_publish_failure_keeps_committed_outbox(monkeypatch):
    db = SimpleNamespace(scalar=AsyncMock(return_value=uuid4()), commit=AsyncMock())
    monkeypatch.setattr(service, "publish_capture", AsyncMock(side_effect=RuntimeError("secret")))
    await service.dispatch_capture(db, uuid4())
    db.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_reconciler_dispatches_queued_sessions_after_restart(monkeypatch):
    session = session_fixture()
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [session]))))
    @asynccontextmanager
    async def context():
        yield db
    monkeypatch.setattr(service, "get_db_context", context)
    publish = AsyncMock()
    monkeypatch.setattr(service, "publish_capture", publish)
    await service.reconcile_captures()
    publish.assert_awaited_once_with(session.id)
    assert session.last_dispatched_at is not None
    await service.reconcile_captures()
    assert publish.await_count == 1


def test_local_write_failure_keeps_existing_artifact(tmp_path):
    storage = LocalStorage(tmp_path)
    storage.save("frame.jpg", io.BytesIO(b"old"))
    class BrokenStream:
        def read(self, size):
            raise OSError("simulated disk read failure")
    with pytest.raises(OSError):
        storage.save("frame.jpg", BrokenStream())
    assert (tmp_path / "frame.jpg").read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [tmp_path / "frame.jpg"]


def test_capture_queue_is_separate_and_bounded():
    assert celery_app.conf.task_default_queue == "prediction"
    assert celery_app.conf.task_routes[run_capture_task.name]["queue"] == "capture"
    assert "api.workers.tasks.capture_task" in celery_app.conf.include
    assert run_capture_task.time_limit == 960
    assert run_capture_task.reject_on_worker_lost is True


@pytest.mark.anyio
async def test_new_sessions_target_accepted_frames_and_auto_compute(runner, monkeypatch):
    from api.services.calibration_quality_service import FrameQuality
    runner.session.selection_version = 1
    runner.session.auto_calibrate = True
    assess = AsyncMock(side_effect=[
        FrameQuality("rejected", "blur", 5, None, None),
        FrameQuality("accepted", None, 100, [[1, 2]], [1, 2]),
        FrameQuality("accepted", None, 100, [[1, 2]], [3, 4]),
    ])
    monkeypatch.setattr(service, "assess_frame_bounded", assess)
    await runner.worker.run()
    assert runner.session.status == "completed"
    assert runner.session.accepted == 2 and runner.session.rejected == 1
    assert runner.session.attempted == runner.session.captured == 3
    assert runner.session.rejection_counts == {"blur": 1}
    assert runner.session.compute_status == "queued"
    assert runner.session.compute_requested_at is not None


@pytest.mark.anyio
async def test_frame_limit_prevents_unbounded_rejected_storage(runner, monkeypatch):
    from api.services.calibration_quality_service import FrameQuality
    runner.session.selection_version = 1
    runner.session.captured = runner.session.attempted = runner.session.rejected = 299
    monkeypatch.setattr(service, "assess_frame_bounded", AsyncMock(return_value=FrameQuality("rejected", "chessboard_not_found", 0, None, None)))
    await runner.worker.run()
    assert runner.session.status == "expired"
    assert runner.session.finish_reason == "frame_limit"
    assert runner.session.accepted == 0
