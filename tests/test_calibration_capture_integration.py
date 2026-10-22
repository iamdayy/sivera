"""Opt-in tests: all configured services must be disposable, never production."""
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.core.config import Settings, settings
from api.core.storage import LocalStorage
from api.models.orm import CalibrationCaptureImage, CalibrationCaptureSession
from api.services.calibration_capture_service import CaptureRunner, utcnow
from tests.test_calibration_capture_service import session_fixture


def database_url():
    url = os.environ.get("SIVERA_TEST_DATABASE_URL")
    if not url:
        pytest.skip("SIVERA_TEST_DATABASE_URL must refer to disposable PostgreSQL")
    return url


def test_capture_migration_constraints_and_cascade(monkeypatch):
    url = database_url()
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    cfg = Config(str(Path(__file__).resolve().parents[1] / "api/alembic.ini"))
    command.upgrade(cfg, "head")
    async def check():
        engine = create_async_engine(url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        camera_id = uuid4()
        try:
            async with engine.begin() as conn:
                await conn.execute(text("INSERT INTO cameras (id,name,resolution_w,resolution_h,fps,metadata) VALUES (:id,'capture-fixture',640,480,25,'{}')"), {"id": camera_id})
            session = session_fixture(camera_id=camera_id)
            async with factory() as db:
                db.add(session)
                await db.commit()
            from sqlalchemy.exc import IntegrityError
            async with factory() as db:
                db.add(session_fixture(camera_id=camera_id))
                with pytest.raises(IntegrityError):
                    await db.commit()
                await db.rollback()
                db.add(session_fixture(camera_id=uuid4()))
                with pytest.raises(IntegrityError):
                    await db.commit()
                await db.rollback()
            async with factory() as db:
                await db.execute(update(CalibrationCaptureSession).where(CalibrationCaptureSession.id == session.id).values(status="stopped"))
                db.add(session_fixture(camera_id=camera_id))
                db.add(CalibrationCaptureImage(id=uuid4(), session_id=session.id, sequence=1, storage_key=str(uuid4()), width=640, height=480, size_bytes=100, captured_at=utcnow()))
                await db.commit()
            async with engine.begin() as conn:
                await conn.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
                assert await conn.scalar(text("SELECT count(*) FROM calibration_capture_sessions WHERE camera_id=:id"), {"id": camera_id}) == 0
                assert await conn.scalar(text("SELECT count(*) FROM calibration_capture_images WHERE session_id=:id"), {"id": session.id}) == 0
        finally:
            await engine.dispose()
    asyncio.run(check())
    command.downgrade(cfg, "003_camera_streams")
    async def downgraded():
        engine = create_async_engine(url)
        try:
            async with engine.connect() as conn:
                assert await conn.scalar(text("SELECT to_regclass('public.calibration_capture_sessions')")) is None
                assert await conn.scalar(text("SELECT to_regclass('public.stream_sessions')")) is not None
        finally:
            await engine.dispose()
    asyncio.run(downgraded())
    command.upgrade(cfg, "head")


def test_real_capture_routes_runner_stop_and_recovery(monkeypatch, tmp_path):
    url = database_url()
    port = os.environ.get("SIVERA_TEST_MEDIAMTX_API_PORT")
    if not port or not os.environ.get("SIVERA_TEST_RTSP_SOURCE"):
        pytest.skip("Requires disposable MediaMTX and a synthetic RTSP publisher")
    from api.core.database import get_db
    from api.core.security import verify_api_key
    from api.main import app
    from api.routes import calibration_captures
    from api.routes.streams import get_mediamtx
    from api.services import calibration_capture_service, stream_service
    from api.services.mediamtx_service import MediaMTXService
    config = Settings(
        _env_file=None, MEDIAMTX_ENABLED=True, MEDIAMTX_HOST="127.0.0.1", MEDIAMTX_API_PORT=int(port),
        MEDIAMTX_RTSP_PORT=int(os.environ.get("SIVERA_TEST_MEDIAMTX_RTSP_PORT", "8554")),
        MEDIAMTX_API_PASSWORD=os.environ["SIVERA_TEST_MEDIAMTX_API_PASSWORD"],
        MEDIAMTX_READ_PASSWORD=os.environ["SIVERA_TEST_MEDIAMTX_READ_PASSWORD"], MEDIAMTX_ALLOWED_RTSP_HOSTS="127.0.0.1",
    )
    async def check():
        engine = create_async_engine(url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        @asynccontextmanager
        async def context():
            async with factory() as db:
                try:
                    yield db
                    await db.commit()
                except Exception:
                    await db.rollback()
                    raise
        async def dependency():
            async with context() as db:
                yield db
        storage = LocalStorage(tmp_path)
        relay = MediaMTXService(config)
        monkeypatch.setattr(calibration_captures, "dispatch_capture", AsyncMock())
        monkeypatch.setattr(calibration_captures, "get_storage", lambda: storage)
        monkeypatch.setattr(stream_service, "get_db_context", context)
        monkeypatch.setattr(calibration_capture_service, "get_db_context", context)
        monkeypatch.setattr(calibration_capture_service, "publish_capture", AsyncMock())
        app.dependency_overrides[get_db] = dependency
        app.dependency_overrides[verify_api_key] = lambda: SimpleNamespace(scopes=["*"], camera_id=None)
        app.dependency_overrides[get_mediamtx] = lambda: relay
        camera_id = None
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/v1/cameras", json={"name": "capture-fixture", "resolution_w": 640, "resolution_h": 480, "rtsp_url": os.environ["SIVERA_TEST_RTSP_SOURCE"]})
                assert response.status_code == 201, response.text
                camera_id = response.json()["id"]
                path = f"/api/v1/cameras/{camera_id}"
                assert (await client.patch(path + "/stream", json={"enabled": True})).status_code == 200
                captures = path + "/calibration-captures"
                assert (await client.get("/health")).json()["database"] == "connected"
                assert captures.replace(camera_id, "{camera_id}") in (await client.get("/openapi.json")).json()["paths"]
                responses = await asyncio.gather(*[client.post(captures, json={"target_images": 2, "interval_sec": 1}) for _ in range(2)])
                assert sorted(r.status_code for r in responses) == [202, 409]
                session_id = next(r.json()["session_id"] for r in responses if r.status_code == 202)
                workers = [CaptureRunner(UUID(session_id), context, storage, config) for _ in range(2)]
                await asyncio.gather(*(worker.run() for worker in workers))
                response = await client.get(captures + "/" + session_id)
                assert response.json()["status"] == "completed", response.text
                assert response.json()["captured"] == response.json()["attempted"] == 2
                response = await client.get(captures + "/" + session_id + "/images")
                assert response.json()["total"] == 2
                for image in response.json()["items"]:
                    assert (image["width"], image["height"]) == (640, 480)
                    response = await client.get(image["download_url"])
                    assert response.status_code == 200
                    assert response.content.startswith(b"\xff\xd8")
                # Stop while a loop is active and retain all completed image writes.
                response = await client.post(captures, json={"target_images": 100, "interval_sec": 1})
                session_id = response.json()["session_id"]
                task = asyncio.create_task(CaptureRunner(UUID(session_id), context, storage, config).run())
                for _ in range(100):
                    progress = (await client.get(captures + "/" + session_id)).json()
                    if progress["captured"]:
                        break
                    await asyncio.sleep(0.2)
                else:
                    pytest.fail("Capture loop did not persist its first image")
                assert (await client.post(captures + "/" + session_id + "/stop")).status_code == 200
                await asyncio.wait_for(task, 5)
                response = await client.get(captures + "/" + session_id)
                assert response.json()["status"] == "stopped"
                assert response.json()["captured"] >= 1
                # Persisted worker loss is reaped without depending on Celery result state.
                response = await client.post(captures, json={})
                session_id = UUID(response.json()["session_id"])
                async with context() as db:
                    await db.execute(update(CalibrationCaptureSession).where(CalibrationCaptureSession.id == session_id).values(
                        status="running", heartbeat_at=utcnow() - timedelta(seconds=200), worker_token=uuid4(),
                    ))
                await calibration_capture_service.reconcile_captures(config)
                assert (await client.get(captures + "/" + str(session_id))).json()["error_code"] == "capture_worker_lost"
                # Deadline in-flight cancels the subprocess and does not become a zero estimate.
                response = await client.post(captures, json={"max_duration_sec": 1, "target_images": 100})
                session_id = UUID(response.json()["session_id"])
                await CaptureRunner(session_id, context, storage, config).run()
                assert (await client.get(captures + "/" + str(session_id))).json()["status"] == "expired"
        finally:
            if camera_id:
                async with engine.begin() as conn:
                    await conn.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
                await relay.delete_path("sivera_" + camera_id)
            app.dependency_overrides.clear()
            await relay.close()
            await engine.dispose()
    asyncio.run(check())


def test_real_celery_capture_queue(monkeypatch, tmp_path):
    url = database_url()
    broker = os.environ.get("SIVERA_TEST_CELERY_BROKER_URL")
    if not broker:
        pytest.skip("Requires a disposable Redis broker and a running capture-only worker")
    from api.workers.celery_app import celery_app
    from api.workers.tasks.capture_task import run_capture_task
    camera_id = uuid4()
    session = session_fixture(camera_id=camera_id, target_images=1)
    shared_root = Path(os.environ["SIVERA_TEST_STORAGE_ROOT"])
    async def prepare():
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("INSERT INTO cameras (id,name,rtsp_url,stream_enabled,resolution_w,resolution_h,fps,metadata) VALUES (:id,'queue-fixture',:source,true,640,480,25,'{}')"), {"id": camera_id, "source": os.environ["SIVERA_TEST_RTSP_SOURCE"]})
            async with async_sessionmaker(engine, expire_on_commit=False)() as db:
                db.add(session)
                await db.commit()
        finally:
            await engine.dispose()
    asyncio.run(prepare())
    # Create the managed relay path before handing off to the real worker.
    import time
    api_url = "http://127.0.0.1:" + os.environ["SIVERA_TEST_MEDIAMTX_API_PORT"]
    with httpx.Client(auth=("sivera_api", os.environ["SIVERA_TEST_MEDIAMTX_API_PASSWORD"])) as client:
        response = client.post(api_url + "/v3/config/paths/add/sivera_" + str(camera_id), json={
            "source": os.environ["SIVERA_TEST_RTSP_SOURCE"], "sourceOnDemand": True, "rtspTransport": "tcp",
        })
        response.raise_for_status()
    monkeypatch.setattr(celery_app.conf, "broker_url", broker)
    try:
        run_capture_task.apply_async(args=[str(session.id)], queue="capture", retry=False)
        async def inspect():
            engine = create_async_engine(url)
            try:
                async with async_sessionmaker(engine, expire_on_commit=False)() as db:
                    row = await db.get(CalibrationCaptureSession, session.id)
                    images = (await db.execute(select(CalibrationCaptureImage).where(CalibrationCaptureImage.session_id == session.id))).scalars().all()
                    return row, images
            finally:
                await engine.dispose()
        for _ in range(100):
            row, images = asyncio.run(inspect())
            if row.status in {"completed", "failed", "expired"}:
                break
            time.sleep(0.3)
        assert row.status == "completed", row.error_code
        assert row.captured == 1 and len(images) == 1
        storage = LocalStorage(shared_root)
        with storage.get(images[0].storage_key) as image:
            assert image.read().startswith(b"\xff\xd8")
        storage.delete(images[0].storage_key)
    finally:
        async def cleanup():
            engine = create_async_engine(url)
            try:
                async with engine.begin() as conn:
                    await conn.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
            finally:
                await engine.dispose()
        asyncio.run(cleanup())
        with httpx.Client(auth=("sivera_api", os.environ["SIVERA_TEST_MEDIAMTX_API_PASSWORD"])) as client:
            client.delete(api_url + "/v3/config/paths/delete/sivera_" + str(camera_id))


def test_real_capture_worker_crash(monkeypatch):
    url = database_url()
    broker = os.environ.get("SIVERA_TEST_CELERY_BROKER_URL")
    if not broker or os.environ.get("SIVERA_TEST_CAPTURE_CRASH") != "true":
        pytest.skip("Requires explicit crash opt-in and a disposable capture worker on this host")
    import signal
    from api.services import calibration_capture_service
    from api.workers.celery_app import celery_app
    from api.workers.tasks.capture_task import run_capture_task
    camera_id = uuid4()
    session = session_fixture(camera_id=camera_id, target_images=100, interval_sec=60)
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    @asynccontextmanager
    async def context():
        async with factory() as db:
            try:
                yield db
                await db.commit()
            except Exception:
                await db.rollback()
                raise
    async def check():
        async with engine.begin() as conn:
            await conn.execute(text("INSERT INTO cameras (id,name,rtsp_url,stream_enabled,resolution_w,resolution_h,fps,metadata) VALUES (:id,'crash-fixture',:source,true,640,480,25,'{}')"), {
                "id": camera_id, "source": os.environ["SIVERA_TEST_RTSP_SOURCE"],
            })
        async with context() as db:
            db.add(session)
        monkeypatch.setattr(calibration_capture_service, "get_db_context", context)
        monkeypatch.setattr(celery_app.conf, "broker_url", broker)
        run_capture_task.apply_async(args=[str(session.id)], queue="capture", retry=False)
        for _ in range(150):
            async with context() as db:
                row = await db.get(CalibrationCaptureSession, session.id)
            if row.captured:
                break
            await asyncio.sleep(0.2)
        else:
            pytest.fail("Disposable worker did not capture a frame before crash test")
        active = await asyncio.to_thread(celery_app.control.inspect(timeout=3).active)
        worker_pid = next(task["worker_pid"] for tasks in active.values() for task in tasks
            if task["name"] == run_capture_task.name and task["args"] == [str(session.id)])
        os.kill(worker_pid, signal.SIGKILL)
        for _ in range(settings.CAPTURE_HEARTBEAT_TIMEOUT_SEC + 10):
            await calibration_capture_service.reconcile_captures()
            async with context() as db:
                row = await db.get(CalibrationCaptureSession, session.id)
            if row.status == "failed":
                break
            await asyncio.sleep(1)
        assert row.error_code == "capture_worker_lost"
        assert row.worker_token is None
        assert row.captured == 1
        # Redelivery after a crash must not restart or duplicate this session.
        run_capture_task.apply_async(args=[str(session.id)], queue="capture", retry=False)
        await asyncio.sleep(2)
        async with context() as db:
            assert await db.scalar(select(func.count()).select_from(CalibrationCaptureImage).where(
                CalibrationCaptureImage.session_id == session.id,
            )) == 1
    from sqlalchemy import func
    api_url = "http://127.0.0.1:" + os.environ["SIVERA_TEST_MEDIAMTX_API_PORT"]
    with httpx.Client(auth=("sivera_api", os.environ["SIVERA_TEST_MEDIAMTX_API_PASSWORD"])) as client:
        client.post(api_url + "/v3/config/paths/add/sivera_" + str(camera_id), json={
            "source": os.environ["SIVERA_TEST_RTSP_SOURCE"], "sourceOnDemand": True, "rtspTransport": "tcp",
        }).raise_for_status()
    async def run():
        try:
            await check()
        finally:
            async with engine.begin() as conn:
                await conn.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
            await engine.dispose()
    try:
        asyncio.run(run())
    finally:
        with httpx.Client(auth=("sivera_api", os.environ["SIVERA_TEST_MEDIAMTX_API_PASSWORD"])) as client:
            client.delete(api_url + "/v3/config/paths/delete/sivera_" + str(camera_id))
