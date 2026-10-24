from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.core.config import Settings
from api.core.database import get_db
from api.core.security import verify_api_key
from api.core.storage import LocalStorage
from api.main import app
from api.models.orm import CalibrationCaptureImage, CalibrationCaptureSession
from api.routes import calibration_captures as routes
from api.routes.streams import get_mediamtx
from api.services.calibration_capture_service import utcnow
from api.services.mediamtx_service import MediaMTXError
from tests.test_calibration_capture_service import jpeg, session_fixture


@pytest.fixture
def setup(monkeypatch, tmp_path):
    session = session_fixture()
    camera = SimpleNamespace(id=session.camera_id, stream_enabled=True, rtsp_url="rtsp://camera.local/live")
    key = SimpleNamespace(scopes=["calibration:read", "calibration:write"], camera_id=None)
    store = {"active": None, "session": session, "image": None, "artifact_camera": camera.id}
    db = MagicMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(one_or_none=lambda: camera, scalars=lambda: SimpleNamespace(all=lambda: [store["image"]] if store["image"] else [])))
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.flush = AsyncMock()
    db.scalar = AsyncMock()
    async def scalar(statement):
        entity = statement.column_descriptions[0]["entity"]
        if entity is CalibrationCaptureImage:
            return store["image"]
        if entity is CalibrationCaptureSession:
            if len(statement._where_criteria) == 1:
                return store["artifact_camera"]
            if "status" in str(statement._where_criteria[1]):
                return store["active"]
            return store["session"]
        return 1 if store["image"] else 0
    db.scalar.side_effect = scalar
    db.add.side_effect = lambda obj: store.update(session=obj, active=obj)
    service = SimpleNamespace(config=Settings(_env_file=None, MEDIAMTX_ALLOWED_RTSP_HOSTS="camera.local"), check_config=MagicMock())
    async def dependency():
        yield db
    app.dependency_overrides[get_db] = dependency
    app.dependency_overrides[verify_api_key] = lambda: key
    app.dependency_overrides[get_mediamtx] = lambda: service
    monkeypatch.setattr(routes, "dispatch_capture", AsyncMock())
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(routes, "get_storage", lambda: storage)
    client = TestClient(app)
    client.headers["X-API-Key"] = "test-key"
    yield SimpleNamespace(client=client, url=f"/api/v1/cameras/{camera.id}/calibration-captures", store=store,
        session=session, camera=camera, db=db, key=key, service=service, storage=storage)
    app.dependency_overrides.clear()
    client.close()


def test_start_is_committed_before_dispatch(setup):
    response = setup.client.post(setup.url, json={})
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "queued"
    assert response.json()["auto_calibrate"] is False
    assert response.json()["session_id"] == str(setup.store["session"].id)
    assert "rtsp" not in response.text and "worker_token" not in response.text
    setup.db.commit.assert_awaited_once()
    routes.dispatch_capture.assert_awaited_once()


def test_start_database_failure_is_safe(setup):
    setup.db.commit.side_effect = RuntimeError("internal credential/path")
    response = setup.client.post(setup.url, json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "capture_database_unavailable"
    assert "credential" not in response.text
    setup.db.rollback.assert_awaited_once()
    routes.dispatch_capture.assert_not_awaited()


def test_duplicate_returns_409_and_expired_allows_start(setup):
    setup.store["active"] = setup.session
    assert setup.client.post(setup.url, json={}).status_code == 409
    setup.session.deadline_at = utcnow() - timedelta(seconds=1)
    assert setup.client.post(setup.url, json={}).status_code == 202
    assert setup.session.status == "expired"


def test_disabled_stream_and_configuration(setup):
    setup.camera.stream_enabled = False
    assert setup.client.post(setup.url, json={}).status_code == 409
    setup.camera.stream_enabled = True
    setup.service.check_config.side_effect = MediaMTXError("stream_service_not_configured")
    assert setup.client.post(setup.url, json={}).status_code == 503
    routes.dispatch_capture.assert_not_awaited()


def test_start_missing_camera_and_invalid_source(setup):
    setup.db.execute.return_value.one_or_none = lambda: None
    assert setup.client.post(setup.url, json={}).status_code == 404
    setup.db.execute.return_value.one_or_none = lambda: setup.camera
    setup.camera.rtsp_url = "http://invalid-scheme.local/live"
    response = setup.client.post(setup.url, json={})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_rtsp_source"
    routes.dispatch_capture.assert_not_awaited()


def test_status_and_worker_lost_recovery(setup):
    setup.session.status = "running"
    setup.session.heartbeat_at = utcnow() - timedelta(seconds=200)
    response = setup.client.get(f"{setup.url}/{setup.session.id}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error_code"] == "capture_worker_lost"


@pytest.mark.parametrize(("status", "expected"), [("queued", "stopped"), ("running", "stopping"), ("completed", "completed")])
def test_stop_persistent_and_idempotent(setup, status, expected):
    setup.session.status = status
    setup.session.heartbeat_at = utcnow()
    response = setup.client.post(f"{setup.url}/{setup.session.id}/stop")
    assert response.status_code == 200
    assert response.json()["status"] == expected
    setup.db.commit.assert_awaited_once()
    assert setup.session.captured == 0


@pytest.mark.parametrize(("method", "suffix", "kwargs"), [
    ("post", "", {"json": {}}), ("get", "/{id}", {}), ("post", "/{id}/stop", {}),
    ("get", "/{id}/images", {}), ("get", "/{id}/images/{id}", {}),
])
def test_scope_and_camera_restrictions(setup, method, suffix, kwargs):
    url = setup.url + suffix.format(id=setup.session.id)
    setup.key.scopes = []
    assert getattr(setup.client, method)(url, **kwargs).status_code == 403
    setup.key.scopes = ["*"]
    setup.key.camera_id = uuid4()
    assert getattr(setup.client, method)(url, **kwargs).status_code == 403


def test_missing_uuid_and_validation(setup):
    setup.store["session"] = None
    assert setup.client.get(f"{setup.url}/{uuid4()}").status_code == 404
    assert setup.client.get(f"{setup.url}/invalid").status_code == 422
    assert setup.client.post(setup.url, json={"auto_calibrate": True}).status_code == 202
    app.dependency_overrides.pop(verify_api_key)
    setup.client.headers.pop("X-API-Key")
    assert setup.client.post(setup.url, json={}).status_code == 422


def test_image_list_download_and_no_generic_file_bypass(setup):
    image = CalibrationCaptureImage(id=uuid4(), session_id=setup.session.id, sequence=1,
        storage_key=f"cameras/{setup.camera.id}/calibration-captures/{setup.session.id}/frame.jpg",
        width=640, height=480, size_bytes=len(jpeg()), captured_at=utcnow(), quality_status="unassessed")
    setup.store["image"] = image
    import io
    setup.storage.save(image.storage_key, io.BytesIO(jpeg()))
    response = setup.client.get(f"{setup.url}/{setup.session.id}/images")
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    item = response.json()["items"][0]
    assert "storage_key" not in item
    download = setup.client.get(item["download_url"])
    assert download.status_code == 200
    assert download.content == jpeg()
    assert download.headers["cache-control"] == "private, no-store"
    setup.key.camera_id = uuid4()
    assert setup.client.get("/files/" + image.storage_key).status_code == 403
    setup.key.camera_id = None
    setup.key.scopes = ["camera:read"]
    assert setup.client.get("/files/" + image.storage_key).status_code == 403


def test_storage_stream_closes_and_errors_safe(setup):
    import io
    from unittest.mock import patch
    stream = io.BytesIO(jpeg())
    image = SimpleNamespace(storage_key="private")
    setup.store["image"] = image
    url = f"{setup.url}/{setup.session.id}/images/{uuid4()}"
    with patch.object(setup.storage, "get", return_value=stream):
        assert setup.client.get(url).status_code == 200
    assert stream.closed
    with patch.object(setup.storage, "get", side_effect=RuntimeError("password filesystem")):
        response = setup.client.get(url)
        assert response.status_code == 503
        assert "password" not in response.text


def test_openapi_capture_contract(setup):
    schema = setup.client.get("/openapi.json").json()
    path = schema["paths"]["/api/v1/cameras/{camera_id}/calibration-captures"]
    assert "202" in path["post"]["responses"]
    assert "application/json" in path["post"]["requestBody"]["content"]
    assert path["post"]["security"] == [{"ApiKeyAuth": []}]
    assert any(p.endswith("/compute") for p in schema["paths"])
    fields = schema["components"]["schemas"]["CaptureCreate"]["properties"]
    assert fields["chessboard_rows"]["minimum"] == 3
    assert fields["auto_calibrate"]["default"] is False
    upload = schema["paths"]["/api/v1/cameras/{camera_id}/calibrations"]["post"]
    ref = upload["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    assert schema["components"]["schemas"][ref]["properties"]["chessboard_rows"]["minimum"] == 3
    invalid = setup.client.post(f"/api/v1/cameras/{setup.camera.id}/calibrations",
                                data={"chessboard_rows": "2"}, files={"files": ("board.jpg", jpeg(), "image/jpeg")})
    assert invalid.status_code == 422


def test_compute_insufficient_active_and_idempotent(setup):
    url = f"{setup.url}/{setup.session.id}/compute"
    assert setup.client.post(url).status_code == 409
    setup.session.status = "completed"
    assert setup.client.post(url).json()["detail"]["code"] == "calibration_insufficient_images"
    setup.session.captured = 12
    response = setup.client.post(url)
    assert response.status_code == 202
    assert response.json()["compute_status"] == "queued"
    requested = setup.session.compute_requested_at
    assert setup.client.post(url).status_code == 202
    assert setup.session.compute_requested_at == requested
    setup.key.camera_id = uuid4()
    assert setup.client.post(url).status_code == 403


def test_preview_auth_and_missing_artifact(setup):
    url = f"{setup.url}/{setup.session.id}/images/{uuid4()}/preview"
    assert setup.client.get(url).status_code == 404
    image = CalibrationCaptureImage(id=uuid4(), session_id=setup.session.id, sequence=1,
        storage_key="missing.jpg", width=640, height=480, size_bytes=100, captured_at=utcnow(), quality_status="accepted")
    setup.store["image"] = image
    assert setup.client.get(url).status_code == 503
    setup.key.camera_id = uuid4()
    assert setup.client.get(url).status_code == 403
