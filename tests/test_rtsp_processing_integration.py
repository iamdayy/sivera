"""Opt-in PostgreSQL, MediaMTX and Celery tests on disposable resources only."""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.core.config import settings
from api.core.storage import LocalStorage
from api.models.orm import Calibration, CalibrationCaptureImage, CalibrationCaptureSession, CameraSnapshot, DepthScale, GroundModel, GroundReference, PredictionJob, PredictionOutput
from api.services.calibration_capture_service import utcnow
from tests.calibration_fixtures import chessboard, projected_corners
from tests.test_calibration_capture_service import session_fixture, jpeg


def url():
    value = os.environ.get("SIVERA_TEST_DATABASE_URL")
    if not value:
        pytest.skip("Requires disposable SIVERA_TEST_DATABASE_URL")
    return value


@asynccontextmanager
async def database():
    engine = create_async_engine(url(), hide_parameters=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    @asynccontextmanager
    async def context():
        async with factory() as db:
            try:
                yield db
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
    try:
        yield context
    finally:
        await engine.dispose()


async def camera(db, camera_id, source=None):
    await db.execute(text("INSERT INTO cameras (id,name,rtsp_url,stream_enabled,resolution_w,resolution_h,fps,metadata) VALUES (:id,'processing-fixture',:source,true,640,480,25,'{}')"), {"id": camera_id, "source": source})


def test_quality_snapshot_migration_legacy_defaults(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", url())
    cfg = Config("api/alembic.ini")
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "004_calibration_captures")
    camera_id, session_id = uuid4(), uuid4()
    async def legacy():
        async with database() as context, context() as db:
            await camera(db, camera_id)
            await db.execute(text("INSERT INTO calibration_capture_sessions (id,camera_id,chessboard_cols,chessboard_rows,square_m,interval_sec,target_images,max_duration_sec,deadline_at) VALUES (:id,:camera,9,6,0.025,2,20,300,now()+interval '5 minutes')"), {"id": session_id, "camera": camera_id})
    asyncio.run(legacy())
    command.upgrade(cfg, "head")
    async def inspect():
        async with database() as context, context() as db:
            row = await db.get(CalibrationCaptureSession, session_id)
            assert row.selection_version == 0 and row.accepted == 0
            assert row.compute_status == "none" and row.auto_calibrate is False
            assert await db.scalar(text("SELECT to_regclass('public.camera_snapshots')")) is not None
            await db.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
    asyncio.run(inspect())


def test_real_compute_inactive_activation_duplicate_and_legacy_selection(monkeypatch, tmp_path):
    from api.core.database import get_db
    from api.core.security import verify_api_key
    from api.main import app
    from api.services.calibration_compute_service import CalibrationComputeRunner
    camera_id = uuid4()
    K, points = projected_corners()
    storage = LocalStorage(tmp_path)
    async def check():
        async with database() as context:
            async def dependency():
                async with context() as db:
                    yield db
            app.dependency_overrides[get_db] = dependency
            app.dependency_overrides[verify_api_key] = lambda: SimpleNamespace(camera_id=None, scopes=["*"])
            try:
                old = Calibration(id=uuid4(), camera_id=camera_id, version=1, k_matrix=K.tolist(), dist_coeffs=[0]*5, is_active=True)
                session = session_fixture(camera_id=camera_id, status="completed", selection_version=1,
                    attempted=12, captured=12, accepted=12, resolution_w=640, resolution_h=480,
                    compute_status="queued", compute_requested_at=utcnow())
                async with context() as db:
                    await camera(db, camera_id)
                    db.add(old)
                    db.add(session)
                    await db.flush()
                    for i, corners in enumerate(points):
                        db.add(CalibrationCaptureImage(id=uuid4(), session_id=session.id, sequence=i+1,
                            storage_key=f"selected/{i}.jpg", width=640, height=480, size_bytes=100,
                            captured_at=utcnow(), quality_status="accepted", corners=corners,
                            pose_descriptor=[0.5,0.5,0.4,0.3,0,1,1]))
                runners = [CalibrationComputeRunner(session.id, context, storage) for _ in range(2)]
                await asyncio.gather(*(runner.run() for runner in runners))
                async with context() as db:
                    row = await db.get(CalibrationCaptureSession, session.id)
                    assert row.compute_status == "done", row.compute_error_code
                    calibration = await db.get(Calibration, row.result_calibration_id)
                    assert calibration.version == 2 and calibration.is_active is False
                    assert (await db.get(Calibration, old.id)).is_active
                    np.testing.assert_allclose(calibration.k_original, K, atol=0.1)
                    assert calibration.k_matrix == calibration.k_undist
                    assert await db.scalar(select(func.count()).select_from(Calibration).where(Calibration.camera_id == camera_id)) == 2
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    path = f"/api/v1/cameras/{camera_id}/calibrations"
                    result = await client.post(f"{path}/{calibration.id}/activate")
                    assert result.status_code == 200, result.text
                    assert (await client.get(path + "/active")).json()["id"] == str(calibration.id)
                # Legacy frames are assessed once; duplicate poses cannot fabricate ten usable views.
                legacy = session_fixture(camera_id=camera_id, status="completed", attempted=10, captured=10,
                    resolution_w=640, resolution_h=480, compute_status="queued", compute_requested_at=utcnow())
                async with context() as db:
                    db.add(legacy)
                    await db.flush()
                    import io
                    for i in range(10):
                        key = f"legacy/{i}.jpg"
                        storage.save(key, io.BytesIO(chessboard()))
                        db.add(CalibrationCaptureImage(id=uuid4(), session_id=legacy.id, sequence=i+1,
                            storage_key=key, width=640, height=480, size_bytes=len(chessboard()), captured_at=utcnow()))
                await CalibrationComputeRunner(legacy.id, context, storage).run()
                async with context() as db:
                    row = await db.get(CalibrationCaptureSession, legacy.id)
                    assert row.compute_status == "failed" and row.compute_error_code == "calibration_insufficient_images"
                    assert row.accepted == 1 and row.rejected == 9
                    assert row.rejection_counts == {"similar_pose": 9}
                    assert (await db.get(Calibration, calibration.id)).is_active
            finally:
                app.dependency_overrides.clear()
                async with context() as db:
                    await db.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
    asyncio.run(check())


def test_real_rtsp_snapshot_and_prediction_pipeline(monkeypatch, tmp_path):
    if not os.environ.get("SIVERA_TEST_MEDIAMTX_API_PORT"):
        pytest.skip("Requires disposable MediaMTX and synthetic publisher")
    from api.core.database import get_db
    from api.core.security import verify_api_key
    from api.main import app
    from api.routes import snapshots, predictions, calibration_captures
    from api.services import snapshot_service, prediction_orchestrator, segment_service, depth_service, volume_service
    from api.services.mediamtx_service import MediaMTXService, desired_path, stream_path
    from api.services.prediction_worker_service import PredictionRunner
    camera_id = uuid4()
    storage = LocalStorage(tmp_path)
    source = os.environ["SIVERA_TEST_RTSP_SOURCE"]
    relay = MediaMTXService(settings)
    K = [[800,0,320],[0,810,240],[0,0,1]]
    async def check():
        async with database() as context:
            async def dependency():
                async with context() as db:
                    yield db
            app.dependency_overrides[get_db] = dependency
            app.dependency_overrides[verify_api_key] = lambda: SimpleNamespace(camera_id=None, scopes=["*"])
            monkeypatch.setattr(snapshots, "get_db_context", context)
            for module in (snapshots, predictions, calibration_captures, prediction_orchestrator, segment_service, depth_service, volume_service):
                monkeypatch.setattr(module, "get_storage", lambda: storage)
            monkeypatch.setattr(predictions.celery_app, "send_task", lambda *a, **kw: None)
            models = tmp_path / "models"
            models.mkdir()
            (models / settings.DEFAULT_SEG_MODEL).touch()
            (models / settings.DEFAULT_DEPTH_MODEL).touch()
            monkeypatch.setattr(settings, "MODELS_DIR", str(models))
            polygon = np.array([[250,200],[280,200],[280,230],[250,230]], np.float32)
            monkeypatch.setattr(segment_service, "predict_instances", lambda *a, **kw: ([{"cls":"Sampah_Berserakan","conf":0.9,"poly":polygon}], {0:"Sampah_Berserakan",1:"Sampah_Overload"}))
            monkeypatch.setattr(depth_service, "predict_depth", lambda image, *a, **kw: np.full(image.shape[:2], 0.5, np.float32))
            try:
                calibration = Calibration(id=uuid4(), camera_id=camera_id, version=1, k_matrix=K, k_original=K, k_undist=K,
                    dist_coeffs=[0]*5, resolution_w=640, resolution_h=480, alpha=0, is_active=True)
                reference = GroundReference(id=uuid4(), camera_id=camera_id, calibration_id=calibration.id, version=1,
                    image_points=[[0,0],[640,0],[640,480],[0,480]], world_points=[[0,0],[6.4,0],[6.4,4.8],[0,4.8]], is_active=True)
                ground = GroundModel(id=uuid4(), camera_id=camera_id, calibration_id=calibration.id, ground_ref_id=reference.id,
                    R_matrix=np.eye(3).tolist(), t_vector=[0,0,0], H_i2w=[[.01,0,0],[0,.01,0],[0,0,1]],
                    H_w2i=[[100,0,0],[0,100,0],[0,0,1]], bev_ppm=20, bev_margin_m=.1)
                scale = DepthScale(id=uuid4(), camera_id=camera_id, ground_model_id=ground.id,
                    depth_model=settings.DEFAULT_DEPTH_MODEL, scale_s=1, offset_b=0, is_active=True)
                async with context() as db:
                    await camera(db, camera_id, source)
                    db.add(calibration)
                    await db.flush()
                    db.add(reference)
                    await db.flush()
                    db.add(ground)
                    await db.flush()
                    db.add(scale)
                await relay.ensure_path(stream_path(camera_id), source, await relay.get_path_config(stream_path(camera_id)))
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    base = f"/api/v1/cameras/{camera_id}"
                    result = await client.post(base + "/snapshots")
                    assert result.status_code == 201, result.text
                    snapshot = result.json()
                    assert (snapshot["width"],snapshot["height"]) == (640,480)
                    assert (await client.get(snapshot["image_url"])).content.startswith(b"\xff\xd8")
                    assert (await client.get(base + "/snapshots/latest")).json()["id"] == snapshot["id"]
                    for source_type, requested in (("latest_rtsp", None), ("rtsp_frame", ["mask", "volume_result_npz"]), ("upload", [])):
                        data = {"source_type": source_type, "volume_cell_m":"0.1", "volume_min_pts_per_cell":"1"}
                        if requested is not None:
                            data["return_outputs"] = json.dumps(requested)
                        files = {"file": ("image.jpg", jpeg(), "image/jpeg")} if source_type == "upload" else None
                        response = await client.post(base + "/predict", data=data, files=files)
                        assert response.status_code == 202, response.text
                        job_id = UUID(response.json()["job_id"])
                        async with context() as db:
                            assert (await db.get(PredictionJob, job_id)).return_outputs == requested
                        await PredictionRunner(job_id, context, storage).run()
                        result = (await client.get(base + "/predict/" + str(job_id))).json()
                        assert result["status"] == "done", result
                        assert result["return_outputs"] == requested
                        assert result["result"]["volume"]["method_grid"]["cell_m"] == .1
                        job_path = base + "/predict/" + str(job_id)
                        outputs = (await client.get(job_path + "/outputs")).json()["outputs"]
                        if requested is None:
                            assert len(outputs) == 13
                        else:
                            assert {output["output_type"] for output in outputs} == set(requested)
                            assert (await client.get(job_path + "/outputs/overlay")).status_code == 404
                            # Internal pipeline files exist, but unregistered outputs stay inaccessible.
                            internal = next(storage.root.glob(f"cameras/{camera_id}/overlay/{job_id}/*.jpg"))
                            assert (await client.get("/files/" + str(internal.relative_to(storage.root)))).status_code == 404
                        if requested:
                            assert (await client.get(job_path + "/outputs/mask")).status_code == 200
                        if source_type != "upload":
                            assert result["source_ts"] is not None
                    async with context() as db:
                        await db.execute(update(CameraSnapshot).where(CameraSnapshot.camera_id == camera_id).values(captured_at=utcnow()-timedelta(seconds=400)))
                    stale = await client.post(base + "/predict", data={"source_type":"latest_rtsp"})
                    assert stale.status_code == 409 and stale.json()["detail"]["code"] == "snapshot_stale"
                    # Source identity invalidates all snapshots after a configured URL change.
                    async with context() as db:
                        await db.execute(text("UPDATE cameras SET rtsp_url=:source WHERE id=:id"), {"source":source+"-changed", "id":camera_id})
                    assert (await client.get(base + "/snapshots/latest")).status_code == 404
            finally:
                app.dependency_overrides.clear()
                await relay.delete_path(stream_path(camera_id))
                await relay.close()
                async with context() as db:
                    await db.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
    asyncio.run(check())


def test_real_prediction_prerequisite_selection():
    from api.services.prediction_orchestrator import submit_prediction_job

    camera_id = uuid4()
    K = [[800, 0, 320], [0, 810, 240], [0, 0, 1]]

    def reference(calibration_id, version, active):
        return GroundReference(id=uuid4(), camera_id=camera_id, calibration_id=calibration_id,
                               version=version, image_points=[], world_points=[], is_active=active)

    def ground(calibration_id, reference_id):
        return GroundModel(id=uuid4(), camera_id=camera_id, calibration_id=calibration_id,
                           ground_ref_id=reference_id, R_matrix=np.eye(3).tolist(), t_vector=[0, 0, 0],
                           H_i2w=np.eye(3).tolist(), H_w2i=np.eye(3).tolist())

    async def check():
        async with database() as context:
            old = Calibration(id=uuid4(), camera_id=camera_id, version=1, k_matrix=K, dist_coeffs=[0]*5)
            current = Calibration(id=uuid4(), camera_id=camera_id, version=2, k_matrix=K, k_original=K,
                                  k_undist=K, resolution_w=640, resolution_h=480, dist_coeffs=[0]*5)
            proper_ref, old_ref = reference(current.id, 1, False), reference(old.id, 2, True)
            proper_ground, old_ground = ground(current.id, proper_ref.id), ground(old.id, old_ref.id)
            mismatched_ground = ground(current.id, old_ref.id)
            proper_scale = DepthScale(id=uuid4(), camera_id=camera_id, ground_model_id=proper_ground.id,
                                      depth_model="fixture", scale_s=1, offset_b=0, is_active=False)
            wrong_scale = DepthScale(id=uuid4(), camera_id=camera_id, ground_model_id=old_ground.id,
                                     depth_model="fixture", scale_s=1, offset_b=0, is_active=True)
            try:
                async with context() as db:
                    await camera(db, camera_id)
                    db.add_all([old, current])
                    await db.flush()
                    db.add_all([proper_ref, old_ref])
                    await db.flush()
                    db.add_all([proper_ground, old_ground, mismatched_ground])
                    await db.flush()
                    db.add_all([proper_scale, wrong_scale])
                async with context() as db:
                    with pytest.raises(ValueError, match="^prediction_requires_calibration$"):
                        await submit_prediction_job(db, camera_id, "upload")
                    await db.execute(update(Calibration).where(Calibration.id == current.id).values(is_active=True))
                async with context() as db:
                    with pytest.raises(ValueError, match="^prediction_requires_ground_model$"):
                        await submit_prediction_job(db, camera_id, "upload")
                    await db.execute(update(GroundReference).where(GroundReference.id == proper_ref.id).values(is_active=True))
                async with context() as db:
                    with pytest.raises(ValueError, match="^prediction_requires_depth_scale$"):
                        await submit_prediction_job(db, camera_id, "upload")
                    await db.execute(update(DepthScale).where(DepthScale.id == proper_scale.id).values(is_active=True))
                async with context() as db:
                    job = await submit_prediction_job(db, camera_id, "rtsp_frame", return_outputs=["mask"])
                    assert (job.calibration_id, job.ground_model_id, job.depth_scale_id) == (current.id, proper_ground.id, proper_scale.id)
                async with context() as db:
                    assert (await db.get(PredictionJob, job.id)).return_outputs == ["mask"]
                    await db.execute(update(Calibration).where(Calibration.camera_id == camera_id).values(is_active=False))
                    await db.execute(update(Calibration).where(Calibration.id == old.id).values(is_active=True))
                async with context() as db:
                    with pytest.raises(ValueError, match="^calibration_intrinsics_missing$"):
                        await submit_prediction_job(db, camera_id, "rtsp_frame")
            finally:
                async with context() as db:
                    await db.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
    asyncio.run(check())


def test_real_calibration_queue_and_prediction_failure(monkeypatch):
    if not os.environ.get("SIVERA_TEST_CELERY_BROKER_URL"):
        pytest.skip("Requires disposable Redis and calibration/prediction worker")
    from api.workers.celery_app import celery_app
    from api.workers.tasks.calibration_task import run_capture_calibration_task
    from api.workers.tasks.predict_task import run_prediction_task
    camera_id = uuid4()
    K, points = projected_corners()
    session = session_fixture(camera_id=camera_id, status="completed", selection_version=1, captured=12, attempted=12,
        accepted=12, resolution_w=640, resolution_h=480, compute_status="queued", compute_requested_at=utcnow())
    async def prepare():
        async with database() as context, context() as db:
            await camera(db, camera_id)
            db.add(session)
            await db.flush()
            for i, corners in enumerate(points):
                db.add(CalibrationCaptureImage(id=uuid4(), session_id=session.id, sequence=i+1, storage_key=f"queue/{camera_id}/{i}.jpg",
                    width=640,height=480,size_bytes=100,captured_at=utcnow(),quality_status="accepted",corners=corners,pose_descriptor=[0.5,0.5,0.4,0.3,0,1,1]))
    asyncio.run(prepare())
    monkeypatch.setattr(celery_app.conf, "broker_url", os.environ["SIVERA_TEST_CELERY_BROKER_URL"])
    async def inspect():
        async with database() as context, context() as db:
            return await db.get(CalibrationCaptureSession, session.id)
    try:
        run_capture_calibration_task.apply_async(args=[str(session.id)], queue="calibration", retry=False)
        for _ in range(150):
            row = asyncio.run(inspect())
            if row.compute_status in {"done","failed"}:
                break
            time.sleep(.2)
        assert row.compute_status == "done", row.compute_error_code
        async def job_prepare():
            async with database() as context, context() as db:
                assert (await db.get(Calibration,row.result_calibration_id)).is_active is False
                ref = GroundReference(id=uuid4(),camera_id=camera_id,calibration_id=row.result_calibration_id,version=1,image_points=[],world_points=[],is_active=True)
                db.add(ref)
                await db.flush()
                gm = GroundModel(id=uuid4(),camera_id=camera_id,calibration_id=row.result_calibration_id,ground_ref_id=ref.id,R_matrix=np.eye(3).tolist(),t_vector=[0,0,0],H_i2w=np.eye(3).tolist(),H_w2i=np.eye(3).tolist())
                db.add(gm)
                await db.flush()
                ds = DepthScale(id=uuid4(),camera_id=camera_id,ground_model_id=gm.id,depth_model="fixture",scale_s=1,offset_b=0)
                db.add(ds)
                await db.flush()
                job = PredictionJob(id=uuid4(),camera_id=camera_id,calibration_id=row.result_calibration_id,ground_model_id=gm.id,depth_scale_id=ds.id,
                    source_type="upload",source_path="deliberately-missing.image",status="pending")
                db.add(job)
                return job.id
        job_id = asyncio.run(job_prepare())
        run_prediction_task.apply_async(args=[str(job_id)], queue="prediction", retry=False)
        async def job_inspect():
            async with database() as context, context() as db:
                return await db.get(PredictionJob,job_id)
        for _ in range(100):
            job = asyncio.run(job_inspect())
            if job.status == "failed":
                break
            time.sleep(.2)
        assert job.status == "failed" and job.error_msg == "prediction_invalid_image"
        assert job.started_at is not None and job.completed_at is not None
    finally:
        async def cleanup():
            async with database() as context, context() as db:
                await db.execute(text("DELETE FROM cameras WHERE id=:id"), {"id":camera_id})
        asyncio.run(cleanup())
