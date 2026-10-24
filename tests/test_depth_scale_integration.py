"""Opt-in checks; synthetic measurements verify wiring, not physical accuracy."""
import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import numpy as np
import pytest
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.core.config import settings
from api.core.database import get_db
from api.core.security import verify_api_key
from api.core.storage import LocalStorage
from api.main import app
from api.models.orm import Camera, Calibration, GroundReference, GroundModel, DepthScale, PredictionJob
from api.routes import ground_refs
from api.services import depth_scale_service, depth_service
from api.services.image_service import encode_image
from api.services.prediction_orchestrator import prepare_prediction_image, submit_prediction_job
from tests.test_depth_scale_service import raw_depth, body


def test_real_cpu_depth_adapter_and_bounded_calibration(monkeypatch, tmp_path):
    model = os.environ.get("SIVERA_TEST_DEPTH_MODEL")
    if not model:
        pytest.skip("Requires existing SIVERA_TEST_DEPTH_MODEL weights; run without network")
    model = Path(model).resolve()
    assert model.is_file()
    monkeypatch.setattr(settings, "MODELS_DIR", str(model.parent))
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(depth_scale_service, "get_storage", lambda: storage)
    camera = SimpleNamespace(id=uuid4(), resolution_w=320, resolution_h=240)
    K = [[250, 0, 160], [0, 250, 120], [0, 0, 1]]
    calibration = SimpleNamespace(k_original=K, k_undist=K, k_matrix=K,
        dist_coeffs=[.1, -.05, 0, 0, 0], resolution_w=320, resolution_h=240)
    image = np.random.default_rng(3).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    storage.save("synthetic.png", io.BytesIO(encode_image(image, ".png")))
    depth = depth_service.predict_depth(prepare_prediction_image(image, calibration, camera, "upload"), str(model))
    assert depth.shape == (240, 320) and np.isfinite(depth).all() and (depth > 0).all()
    ordered = np.argsort(depth.ravel())
    positions = ordered[[len(ordered)//10, len(ordered)//2, len(ordered)*9//10]]
    measurements = [{"x": int(index % 320), "y": int(index // 320), "distance_m": float(depth.ravel()[index] * .5 + .2)} for index in positions]
    result = asyncio.run(depth_scale_service.compute_depth_scale(None, camera, calibration, None,
        "synthetic.png", model.name, measurements))
    assert result["scale_s"] == pytest.approx(.5, abs=1e-4)
    assert result["offset_b"] == pytest.approx(.2, abs=1e-4)
    assert result["max_residual_m"] < 1e-4


def test_postgres_create_activate_concurrency_and_prediction_pins(monkeypatch, tmp_path):
    url = os.environ.get("SIVERA_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Requires a migrated disposable SIVERA_TEST_DATABASE_URL")
    camera_id, calibration_id, reference_id, ground_id = (uuid4() for _ in range(4))
    storage = LocalStorage(tmp_path / "storage")
    for module in (ground_refs, depth_scale_service):
        monkeypatch.setattr(module, "get_storage", lambda: storage)
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26n-depth.pt").touch()
    monkeypatch.setattr(settings, "MODELS_DIR", str(models))
    async def inference(mode, payload, timeout):
        from api.schemas.ground_ref import DepthScaleCreate
        await asyncio.sleep(.02)
        return depth_scale_service.fit_depth_scale(raw_depth(), DepthScaleCreate.model_validate(payload["body"]))
    monkeypatch.setattr(depth_scale_service, "run_vision", inference)
    key = SimpleNamespace(camera_id=None, scopes=["ground_ref:read", "ground_ref:write"])
    previous = app.dependency_overrides.copy()
    async def check():
        engine = create_async_engine(url, hide_parameters=True)
        factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
        @asynccontextmanager
        async def context():
            async with factory() as db:
                try:
                    yield db
                    await db.commit()
                except BaseException:
                    await db.rollback()
                    raise
        async def dependency():
            async with context() as db:
                yield db
        app.dependency_overrides[get_db] = dependency
        app.dependency_overrides[verify_api_key] = lambda: key
        K = [[50, 0, 32], [0, 50, 24], [0, 0, 1]]
        prefix = f"/api/v1/cameras/{camera_id}/ground-models/{ground_id}/depth-scales"
        try:
            async with context() as db:
                db.add(Camera(id=camera_id, name="depth-disposable", resolution_w=64, resolution_h=48))
                await db.flush()
                db.add(Calibration(id=calibration_id, camera_id=camera_id, version=1, k_original=K, k_undist=K,
                    k_matrix=K, dist_coeffs=[0]*5, resolution_w=64, resolution_h=48, is_active=True))
                await db.flush()
                db.add(GroundReference(id=reference_id, camera_id=camera_id, calibration_id=calibration_id,
                    version=1, image_points=[[0, 0], [63, 0], [63, 47], [0, 47]],
                    world_points=[[0, 0], [1, 0], [1, 1], [0, 1]], is_active=True))
                await db.flush()
                db.add(GroundModel(id=ground_id, camera_id=camera_id, calibration_id=calibration_id,
                    ground_ref_id=reference_id, R_matrix=np.eye(3).tolist(), t_vector=[0, 0, 1],
                    H_i2w=np.eye(3).tolist(), H_w2i=np.eye(3).tolist()))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                assert (await client.get(prefix + "/active")).status_code == 404
                async with context() as db:
                    with pytest.raises(ValueError, match="prediction_requires_depth_scale"):
                        await submit_prediction_job(db, camera_id, "upload")
                uploaded = await client.post(prefix + "/test-images", files={"file": ("image.png", encode_image(np.zeros((48, 64, 3), np.uint8), ".png"))})
                assert uploaded.status_code == 201, uploaded.text
                request = body(test_image=uploaded.json()["test_image"]).model_dump()
                created = []
                for _ in range(2):
                    response = await client.post(prefix, json=request)
                    assert response.status_code == 201, response.text
                    created.append(response.json()["id"])
                results = await asyncio.gather(*(client.post(prefix + f"/{item}/activate") for item in created * 3))
                assert all(response.status_code == 200 for response in results), [response.text for response in results]
                async with context() as db:
                    assert await db.scalar(select(func.count()).select_from(DepthScale).where(DepthScale.ground_model_id == ground_id, DepthScale.is_active == True)) == 1
                    active = await db.scalar(select(DepthScale).where(DepthScale.ground_model_id == ground_id, DepthScale.is_active == True))
                    job = await submit_prediction_job(db, camera_id, "upload")
                    job_id, pinned = job.id, job.depth_scale_id
                    assert pinned == active.id and job.calibration_id == calibration_id and job.ground_model_id == ground_id
                from uuid import UUID
                other = next(item for item in created if UUID(item) != pinned)
                assert (await client.post(prefix + f"/{other}/activate")).status_code == 200
                async with context() as db:
                    assert (await db.get(PredictionJob, job_id)).depth_scale_id == pinned
                key.camera_id = uuid4()
                assert (await client.get(prefix)).status_code == 403
                assert (await client.post(prefix + f"/{other}/activate")).status_code == 403
                assert (await client.get(uploaded.json()["image_url"])).status_code == 403
                key.camera_id = None
                # Concurrent create is bounded; successful writes still leave one active scale.
                results = await asyncio.gather(*(client.post(prefix, json=request) for _ in range(3)))
                assert any(response.status_code == 201 for response in results)
                assert all(response.status_code in {201, 503} for response in results)
                async with context() as db:
                    assert await db.scalar(select(func.count()).select_from(DepthScale).where(DepthScale.ground_model_id == ground_id, DepthScale.is_active == True)) == 1
        finally:
            from sqlalchemy import delete
            async with context() as db:
                await db.execute(delete(Camera).where(Camera.id == camera_id))
            app.dependency_overrides.clear()
            app.dependency_overrides.update(previous)
            await engine.dispose()
    asyncio.run(check())
