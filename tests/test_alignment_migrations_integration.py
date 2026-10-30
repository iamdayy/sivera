"""Opt-in schema downgrade/upgrade and access checks on a disposable PostGIS database."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from fastapi import HTTPException
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError

from api.core.config import settings
from api.models.orm import Calibration, Camera, DepthScale, GroundModel, GroundReference, PreviewArtifact
from api.services.depth_model_service import validate_scale_model
from api.services.prediction_access_service import authorize_artifact
from tests.test_rtsp_processing_integration import camera, database, url


def test_legacy_identity_preview_registration_and_parent_cascades(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", url())
    cfg = Config(str(Path(__file__).resolve().parents[1] / "api/alembic.ini"))
    camera_id, calib_id, ground_id, scale_id = (uuid4() for _ in range(4))
    source = f"cameras/{camera_id}/calib_images/legacy.png"
    preview = f"cameras/{camera_id}/calib_preview/legacy.jpg"
    bev = f"cameras/{camera_id}/bev_image/preview.jpg"
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "005_capture_quality_snapshots")

    async def seed():
        async with database() as context, context() as db:
            await camera(db, camera_id)
            db.add(Calibration(id=calib_id, camera_id=camera_id, version=1, k_matrix=np.eye(3).tolist(),
                               dist_coeffs=[0]*5, calib_images=[source], is_active=True))
            await db.flush()
            ref = GroundReference(id=uuid4(), camera_id=camera_id, calibration_id=calib_id,
                                  version=1, image_points=[], world_points=[], is_active=True)
            db.add(ref)
            await db.flush()
            db.add(GroundModel(id=ground_id, camera_id=camera_id, calibration_id=calib_id, ground_ref_id=ref.id,
                R_matrix=np.eye(3).tolist(), t_vector=[0, 0, 1], H_i2w=np.eye(3).tolist(), H_w2i=np.eye(3).tolist()))
            await db.flush()
            await db.execute(text("""INSERT INTO depth_scales
                (id, camera_id, ground_model_id, depth_model, scale_s, offset_b, clip_min_m, clip_max_m, is_active)
                VALUES (:id, :camera, :ground, 'legacy.pt', 1, 0, .1, 50, true)"""),
                {"id": scale_id, "camera": camera_id, "ground": ground_id})

    async def inspect():
        async with database() as context, context() as db:
            scale = await db.get(DepthScale, scale_id)
            assert scale.model_identity is None and scale.fit_provenance is None
            assert scale.is_active and scale.scale_s == 1
            with pytest.raises(ValueError, match="depth_scale_identity_missing"):
                validate_scale_model(scale)
            key = SimpleNamespace(camera_id=camera_id, scopes=["calibration:read"])
            await authorize_artifact(db, source, key)
            for path in (preview, source + "-orphan"):
                with pytest.raises(HTTPException) as exc:
                    await authorize_artifact(db, path, key)
                assert exc.value.status_code == 404
            db.add_all([PreviewArtifact(storage_key=preview, calibration_id=calib_id),
                        PreviewArtifact(storage_key=bev, ground_model_id=ground_id)])
            await db.flush()
            await authorize_artifact(db, preview, key)
            with pytest.raises(HTTPException) as exc:
                await authorize_artifact(db, bev, key)
            assert exc.value.status_code == 403
            key.scopes = ["ground_ref:read"]
            await authorize_artifact(db, bev, key)
            key.camera_id = uuid4()
            with pytest.raises(HTTPException) as exc:
                await authorize_artifact(db, bev, key)
            assert exc.value.status_code == 403
            key.camera_id = camera_id
            for parents in ({}, {"calibration_id": calib_id, "ground_model_id": ground_id}):
                with pytest.raises(IntegrityError):
                    async with db.begin_nested():
                        db.add(PreviewArtifact(storage_key=f"invalid/{uuid4()}", **parents))
                        await db.flush()
            await db.execute(delete(GroundModel).where(GroundModel.id == ground_id))
            assert await db.scalar(select(PreviewArtifact.storage_key).where(PreviewArtifact.storage_key == bev)) is None
            with pytest.raises(HTTPException) as exc:
                await authorize_artifact(db, bev, key)
            assert exc.value.status_code == 404
            await db.execute(delete(Calibration).where(Calibration.id == calib_id))
            assert await db.scalar(select(PreviewArtifact.storage_key).where(PreviewArtifact.storage_key == preview)) is None
            key.scopes = ["calibration:read"]
            with pytest.raises(HTTPException) as exc:
                await authorize_artifact(db, source, key)
            assert exc.value.status_code == 404

    try:
        asyncio.run(seed())
        command.upgrade(cfg, "head")
        asyncio.run(inspect())
    finally:
        command.upgrade(cfg, "head")
        async def cleanup():
            async with database() as context, context() as db:
                await db.execute(delete(Camera).where(Camera.id == camera_id))
        asyncio.run(cleanup())
