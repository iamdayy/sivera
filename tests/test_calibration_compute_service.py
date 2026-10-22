import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager
from uuid import uuid4

from api.services import calibration_compute_service as service
from api.services.calibration_capture_service import utcnow
from api.workers.celery_app import celery_app
from api.workers.tasks.calibration_task import run_capture_calibration_task
from tests.test_calibration_capture_service import session_fixture


def test_compute_queue_timeout_and_worker_loss_fence():
    queued = session_fixture(compute_status="queued", compute_requested_at=utcnow() - timedelta(seconds=200))
    service.expire_computation(queued)
    assert queued.compute_status == "failed"
    assert queued.compute_error_code == "calibration_queue_timeout"
    running = session_fixture(compute_status="running", compute_started_at=utcnow() - timedelta(seconds=400))
    service.expire_computation(running)
    assert running.compute_error_code == "calibration_compute_timeout"
    assert running.compute_token is None


def test_compute_outbox_retries_after_queue_failure(monkeypatch):
    session = session_fixture(compute_status="queued", compute_requested_at=utcnow())
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [session]))))
    @asynccontextmanager
    async def context():
        yield db
    publish = AsyncMock(side_effect=RuntimeError("private/password"))
    monkeypatch.setattr(service, "get_db_context", context)
    monkeypatch.setattr(service, "publish_computation", publish)
    asyncio.run(service.reconcile_computations())
    assert session.compute_status == "queued"
    assert session.compute_dispatched_at is not None
    asyncio.run(service.reconcile_computations())
    assert publish.await_count == 1
    session.compute_dispatched_at = utcnow() - timedelta(seconds=16)
    asyncio.run(service.reconcile_computations())
    assert publish.await_count == 2


def test_compute_queue_isolated():
    assert celery_app.conf.task_routes[run_capture_calibration_task.name]["queue"] == "calibration"
    assert run_capture_calibration_task.time_limit == 660
    assert run_capture_calibration_task.reject_on_worker_lost


def test_old_compute_token_and_timeout_prevent_persistence():
    session = session_fixture(compute_status="running", compute_started_at=utcnow(), compute_token=uuid4())
    db = SimpleNamespace(scalar=AsyncMock(return_value=session))
    @asynccontextmanager
    async def context():
        yield db
    runner = service.CalibrationComputeRunner(session.id, context, None)
    assert asyncio.run(runner.active()) is False
    session.compute_token = runner.token
    assert asyncio.run(runner.active()) is True
    session.compute_started_at = utcnow() - timedelta(seconds=400)
    assert asyncio.run(runner.active()) is False
    assert session.compute_status == "failed" and session.compute_token is None


def test_compute_source_change_fences_worker():
    session = session_fixture(compute_status="running", compute_started_at=utcnow(), source_hash="old-source")
    db = SimpleNamespace(scalar=AsyncMock(side_effect=[session, "rtsp://camera.local/new"]))
    @asynccontextmanager
    async def context():
        yield db
    runner = service.CalibrationComputeRunner(session.id, context, None)
    session.compute_token = runner.token
    assert asyncio.run(runner.active()) is False
    assert session.compute_error_code == "calibration_source_changed"
    assert session.compute_token is None
