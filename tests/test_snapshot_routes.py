from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.core.database import get_db
from api.core.security import verify_api_key
from api.main import app
from api.models.orm import CameraSnapshot
from api.routes import snapshots
from api.services.calibration_capture_service import utcnow
from api.services.rtsp_capture_service import CaptureError


@pytest.fixture
def setup(monkeypatch):
    camera_id = uuid4()
    snapshot = CameraSnapshot(id=uuid4(), camera_id=camera_id, source_hash="private-hash", storage_key="private-key",
                              width=640, height=480, size_bytes=100, captured_at=utcnow())
    key = SimpleNamespace(camera_id=None, scopes=["camera:read", "camera:write"])
    db = MagicMock(scalar=AsyncMock(return_value=snapshot), execute=AsyncMock(
        return_value=SimpleNamespace(one_or_none=lambda: SimpleNamespace(rtsp_url="rtsp://camera/live", stream_enabled=True))))
    async def dependency():
        yield db
    app.dependency_overrides[get_db] = dependency
    app.dependency_overrides[verify_api_key] = lambda: key
    monkeypatch.setattr(snapshots, "capture_snapshot", AsyncMock(return_value=snapshot))
    monkeypatch.setattr(snapshots, "latest_snapshot", AsyncMock(return_value=snapshot))
    monkeypatch.setattr(snapshots, "get_storage", lambda: None)
    monkeypatch.setattr(snapshots, "require_rtsp_camera", lambda _: "private-hash")
    monkeypatch.setattr(snapshots, "image_response", AsyncMock(return_value={"image": True}))
    client = TestClient(app)
    yield SimpleNamespace(client=client, key=key, db=db, snapshot=snapshot, url=f"/api/v1/cameras/{camera_id}/snapshots")
    client.close()
    app.dependency_overrides.clear()


def test_snapshot_response_and_freshness_validation(setup):
    response = setup.client.post(setup.url)
    assert response.status_code == 201
    assert response.json()["image_url"].endswith(f"/{setup.snapshot.id}/image")
    assert "private" not in response.text
    assert setup.client.get(setup.url + "/latest?max_frame_age_sec=301").status_code == 422
    snapshots.latest_snapshot.side_effect = CaptureError("snapshot_stale")
    response = setup.client.get(setup.url + "/latest")
    assert response.status_code == 409 and response.json()["detail"]["code"] == "snapshot_stale"


@pytest.mark.parametrize(("method", "suffix"), [("post", ""), ("get", "/latest"), ("get", "/id/image")])
def test_snapshot_camera_scope_and_key_required(setup, method, suffix):
    suffix = suffix.replace("id", str(setup.snapshot.id))
    setup.key.camera_id = uuid4()
    assert getattr(setup.client, method)(setup.url + suffix).status_code == 403
    setup.key.camera_id = None
    setup.key.scopes = ["predict:read", "predict:write"]
    assert getattr(setup.client, method)(setup.url + suffix).status_code == 403
    app.dependency_overrides.pop(verify_api_key)
    assert getattr(setup.client, method)(setup.url + suffix).status_code == 422


def test_snapshot_image_missing_and_failure_sanitization(setup):
    setup.db.scalar.return_value = None
    assert setup.client.get(f"{setup.url}/{setup.snapshot.id}/image").status_code == 404
    snapshots.capture_snapshot.side_effect = RuntimeError("rtsp://password@private/path")
    response = setup.client.post(setup.url)
    assert response.status_code == 503
    assert "password" not in response.text
