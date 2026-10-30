"""Preview generation and authenticated downloads on disposable PostgreSQL/storage."""
import asyncio
import io
from pathlib import Path
from uuid import uuid4

import cv2
import httpx
import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, func, select

from api.core.config import settings
from api.core.database import get_db
from api.core.security import hash_key
from api.core.storage import LocalStorage
from api import main
from api.models.orm import ApiKey, Calibration, Camera, PreviewArtifact
from api.routes import calibrations
from tests.test_rtsp_processing_integration import camera, database, url


@pytest.mark.parametrize("legacy_cached", [False, True])
def test_preview_registration_download_cache_and_access(monkeypatch, tmp_path, legacy_cached):
    monkeypatch.setattr(settings, "DATABASE_URL", url())
    monkeypatch.setattr(settings, "VALID_API_KEYS", "")
    command.upgrade(Config(str(Path(__file__).resolve().parents[1] / "api/alembic.ini")), "head")
    camera_id, other_camera_id, calibration_id = (uuid4() for _ in range(3))
    source = f"cameras/{camera_id}/calib_images/source.png"
    preview = f"cameras/{camera_id}/calibrations/{calibration_id}/preview.jpg"
    path = f"/api/v1/cameras/{camera_id}/calibrations/{calibration_id}/undistort-preview"
    K = [[50, 0, 32], [0, 50, 24], [0, 0, 1]]
    storage = LocalStorage(tmp_path)
    image = np.random.default_rng(3).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    storage.save(source, io.BytesIO(encoded.tobytes()), "image/png")
    if legacy_cached:
        storage.save(preview, io.BytesIO(b"legacy unregistered bytes"))
    for module in (main, calibrations):
        monkeypatch.setattr(module, "get_storage", lambda: storage)
    read_key, wrong_scope_key, foreign_key = (str(uuid4()) for _ in range(3))
    keys = [ApiKey(id=uuid4(), key_hash=hash_key(value), name="preview-test", camera_id=owner, scopes=scopes)
            for value, owner, scopes in [(read_key, camera_id, ["calibration:read"]),
                                        (wrong_scope_key, camera_id, ["predict:read"]),
                                        (foreign_key, other_camera_id, ["calibration:read"])]]

    async def check():
        async with database() as context:
            async def dependency():
                async with context() as db:
                    yield db
            previous = main.app.dependency_overrides.copy()
            main.app.dependency_overrides[get_db] = dependency
            try:
                async with context() as db:
                    await camera(db, camera_id)
                    await camera(db, other_camera_id)
                    db.add(Calibration(id=calibration_id, camera_id=camera_id, version=1,
                        k_original=K, k_undist=K, k_matrix=K, dist_coeffs=[.3, -.1, 0, 0, 0],
                        resolution_w=64, resolution_h=48, calib_images=[source], is_active=False))
                    db.add_all(keys)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                    headers = {"X-API-Key": read_key}
                    assert (await client.get(f"/files/{preview}", headers=headers)).status_code == 404
                    responses = await asyncio.gather(*(client.get(path, headers=headers) for _ in range(2)))
                    assert [response.status_code for response in responses] == [200, 200]
                    metadata = responses[0].json()
                    assert metadata == responses[1].json()
                    assert metadata["original_size"] == metadata["undistorted_size"] == [64, 48]
                    assert metadata["preview_url"] == f"/files/{preview}"
                    downloaded = await client.get(metadata["preview_url"], headers=headers)
                    assert downloaded.status_code == 200
                    assert downloaded.headers["content-type"] == "image/jpeg"
                    decoded = cv2.imdecode(np.frombuffer(downloaded.content, np.uint8), cv2.IMREAD_COLOR)
                    assert decoded is not None and decoded.shape == image.shape
                    async with context() as db:
                        assert await db.scalar(select(func.count()).select_from(PreviewArtifact).where(
                            PreviewArtifact.storage_key == preview, PreviewArtifact.calibration_id == calibration_id)) == 1
                        assert (await db.get(Calibration, calibration_id)).is_active is False

                    cached = await client.get(path, headers=headers)
                    assert cached.status_code == 200 and cached.json() == metadata
                    assert (await client.get(metadata["preview_url"], headers=headers)).content == downloaded.content
                    assert storage.delete(preview)
                    assert (await client.get(path, headers=headers)).status_code == 200
                    assert (await client.get(metadata["preview_url"], headers=headers)).content == downloaded.content

                    for target in (path, metadata["preview_url"]):
                        assert (await client.get(target)).status_code == 422
                        assert (await client.get(target, headers={"X-API-Key": "invalid-test-key"})).status_code == 401
                        for denied in (wrong_scope_key, foreign_key):
                            assert (await client.get(target, headers={"X-API-Key": denied})).status_code == 403
                    async with context() as db:
                        await db.execute(delete(Calibration).where(Calibration.id == calibration_id))
                    assert storage.exists(preview)
                    assert (await client.get(metadata["preview_url"], headers=headers)).status_code == 404
                    assert (await client.get(path, headers=headers)).status_code == 404
            finally:
                main.app.dependency_overrides.clear()
                main.app.dependency_overrides.update(previous)
                async with context() as db:
                    await db.execute(delete(ApiKey).where(ApiKey.id.in_([key.id for key in keys])))
                    await db.execute(delete(Camera).where(Camera.id.in_([camera_id, other_camera_id])))
    asyncio.run(check())
