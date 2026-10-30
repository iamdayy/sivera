import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from api.core.storage import LocalStorage
from api.models.orm import PreviewArtifact
from api.routes import calibrations


@pytest.fixture
def preview(monkeypatch, tmp_path):
    camera_id, calibration_id = uuid4(), uuid4()
    source = f"cameras/{camera_id}/calib_images/source.png"
    key = f"cameras/{camera_id}/calibrations/{calibration_id}/preview.jpg"
    K = [[50, 0, 32], [0, 50, 24], [0, 0, 1]]
    calibration = SimpleNamespace(id=calibration_id, camera_id=camera_id, calib_images=[source],
        k_original=K, k_matrix=K, k_undist=K, dist_coeffs=[.3, -.1, 0, 0, 0])
    image = np.random.default_rng(2).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    storage = LocalStorage(tmp_path)
    storage.save(source, io.BytesIO(encoded.tobytes()), "image/png")
    monkeypatch.setattr(calibrations, "get_storage", lambda: storage)
    db = MagicMock(execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: calibration)),
                   scalar=AsyncMock(return_value=None), commit=AsyncMock())
    principal = SimpleNamespace(camera_id=camera_id, scopes=["calibration:read"])
    def request():
        return asyncio.run(calibrations.get_undistort_preview(camera_id, calibration_id, db, principal))
    return SimpleNamespace(storage=storage, key=key, source=source, calibration=calibration,
                           db=db, request=request, image=image, principal=principal)


@pytest.mark.parametrize("cached,registered", [(False, False), (True, False), (False, True)])
def test_preview_generation_repairs_missing_file_or_registration(preview, cached, registered):
    if cached:
        preview.storage.save(preview.key, io.BytesIO(b"unregistered legacy preview"))
    if registered:
        preview.db.scalar.return_value = preview.key
    result = preview.request()
    assert result.preview_url == f"/files/{preview.key}"
    assert result.original_size == result.undistorted_size == [64, 48]
    with preview.storage.get(preview.key) as stream:
        decoded = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == preview.image.shape
    if registered:
        preview.db.add.assert_not_called()
    else:
        row = preview.db.add.call_args.args[0]
        assert isinstance(row, PreviewArtifact)
        assert row.storage_key == preview.key
        assert row.calibration_id == preview.calibration.id
        assert row.ground_model_id is None
    preview.db.commit.assert_awaited_once()


def test_registered_cached_preview_is_reused(preview, monkeypatch):
    preview.db.scalar.return_value = preview.key
    ok, encoded = cv2.imencode(".jpg", preview.image)
    assert ok
    saved = encoded.tobytes()
    preview.storage.save(preview.key, io.BytesIO(saved), "image/jpeg")
    undistort = MagicMock(side_effect=AssertionError("cached preview should not be recomputed"))
    monkeypatch.setattr(cv2, "undistort", undistort)
    assert preview.request().preview_url == f"/files/{preview.key}"
    with preview.storage.get(preview.key) as stream:
        assert stream.read() == saved
    undistort.assert_not_called()
    preview.db.add.assert_not_called()


@pytest.mark.parametrize("failure", ["encode", "save", "commit"])
def test_failed_preview_never_returns_a_success_url(preview, monkeypatch, failure):
    if failure == "encode":
        monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    elif failure == "save":
        monkeypatch.setattr(preview.storage, "save", MagicMock(side_effect=OSError("storage unavailable")))
    else:
        preview.db.commit.side_effect = RuntimeError("database unavailable")
    with pytest.raises((HTTPException, OSError, RuntimeError)):
        preview.request()
    if failure != "commit":
        preview.db.add.assert_not_called()
        preview.db.commit.assert_not_awaited()


def test_camera_denial_happens_before_preview_access(preview):
    preview.principal.camera_id = uuid4()
    with pytest.raises(HTTPException) as exc:
        preview.request()
    assert exc.value.status_code == 403
    preview.db.execute.assert_not_awaited()
    preview.db.add.assert_not_called()


def test_legacy_intrinsics_are_not_invented_to_register_preview(preview):
    preview.calibration.k_original = None
    with pytest.raises(HTTPException) as exc:
        preview.request()
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "calibration_intrinsics_missing"
    preview.db.add.assert_not_called()
