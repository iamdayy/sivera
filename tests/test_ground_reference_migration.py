"""Additive source migration on an opt-in disposable PostgreSQL database."""
import asyncio
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from api.core.config import settings
from tests.test_rtsp_processing_integration import camera, database, url


def test_source_migration_preserves_legacy_references(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", url())
    cfg = Config(str(Path(__file__).resolve().parents[1] / "api/alembic.ini"))
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "007_preview_artifacts")
    camera_id, calibration_id, reference_id = (uuid4() for _ in range(3))

    async def seed():
        async with database() as context, context() as db:
            await camera(db, camera_id)
            await db.execute(text("INSERT INTO calibrations (id,camera_id,version,k_matrix,dist_coeffs,is_active) VALUES (:id,:camera,1,'[[50,0,32],[0,50,24],[0,0,1]]','[0,0,0,0,0]',true)"),
                             {"id": calibration_id, "camera": camera_id})
            await db.execute(text("INSERT INTO ground_references (id,camera_id,calibration_id,version,image_points,world_points,world_units,is_active,description) VALUES (:id,:camera,:calibration,1,'[[2,2],[40,2],[40,40],[2,40]]','[[0,0],[2,0],[2,2],[0,2]]','meter',true,'legacy reference')"),
                             {"id": reference_id, "camera": camera_id, "calibration": calibration_id})

    async def verify(upgraded):
        async with database() as context, context() as db:
            row = (await db.execute(text("SELECT * FROM ground_references WHERE id=:id"), {"id": reference_id})).mappings().one()
            assert row["is_active"] and row["version"] == 1 and row["description"] == "legacy reference"
            assert row["world_points"] == [[0, 0], [2, 0], [2, 2], [0, 2]]
            if upgraded:
                assert row["source_id"] is None
                assert await db.scalar(text("SELECT count(*) FROM ground_reference_sources")) == 0
            else:
                assert "source_id" not in row

    async def cleanup():
        async with database() as context, context() as db:
            await db.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})

    asyncio.run(seed())
    try:
        command.upgrade(cfg, "head")
        asyncio.run(verify(True))
        command.downgrade(cfg, "007_preview_artifacts")
        asyncio.run(verify(False))
        command.upgrade(cfg, "head")
        asyncio.run(verify(True))
    finally:
        asyncio.run(cleanup())
