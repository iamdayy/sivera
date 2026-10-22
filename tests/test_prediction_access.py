import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from api.services.prediction_access_service import authorize_artifact


def test_ground_preview_compatibility_and_prediction_scope():
    camera_id, job_id = uuid4(), uuid4()
    db = SimpleNamespace(scalar=AsyncMock(return_value=job_id))
    key = SimpleNamespace(camera_id=camera_id, scopes=["ground_ref:read"])
    for kind in ("bev_image", "bev_overlay", "bev_mask", "depth_scale_npz"):
        asyncio.run(authorize_artifact(db, f"cameras/{camera_id}/{kind}/preview.jpg", key))
    db.scalar.assert_not_awaited()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(authorize_artifact(db, f"cameras/{camera_id}/bev_image/{job_id}/output.jpg", key))
    assert exc.value.status_code == 403
    key.scopes = ["predict:read"]
    asyncio.run(authorize_artifact(db, f"cameras/{camera_id}/bev_image/{job_id}/output.jpg", key))
    db.scalar.return_value = None
    with pytest.raises(HTTPException) as exc:
        asyncio.run(authorize_artifact(db, f"cameras/{camera_id}/mask/{job_id}/orphan.png", key))
    assert exc.value.status_code == 404


@pytest.mark.parametrize("path", ["/cameras/{id}/mask/a", "cameras//{id}/mask/a", "cameras/{id}/../mask/a", "cameras/{id}/./mask/a", "cameras\\{id}\\mask\\a"])
def test_noncanonical_artifact_keys_cannot_bypass_checks(path):
    db = SimpleNamespace(scalar=AsyncMock(return_value=uuid4()))
    key = SimpleNamespace(camera_id=None, scopes=["*"])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(authorize_artifact(db, path.format(id=uuid4()), key))
    assert exc.value.status_code == 404
    db.scalar.assert_not_awaited()


def test_snapshot_files_need_camera_read_and_matching_camera():
    camera_id = uuid4()
    db = SimpleNamespace(scalar=AsyncMock(return_value=uuid4()))
    key = SimpleNamespace(camera_id=None, scopes=["predict:read"])
    path = f"cameras/{camera_id}/snapshots/frame.jpg"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(authorize_artifact(db, path, key))
    assert exc.value.status_code == 403
    key.scopes = ["camera:read"]
    asyncio.run(authorize_artifact(db, path, key))
    key.camera_id = uuid4()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(authorize_artifact(db, path, key))
    assert exc.value.status_code == 403
