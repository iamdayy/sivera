from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from api.core.config import Settings
from api.core.database import get_db
from api.core.security import hash_key, verify_api_key
from api.main import app
from api.models.orm import StreamSession
from api.routes import streams, cameras
from api.services.mediamtx_service import MediaMTXError, desired_path, stream_path


@pytest.fixture
def setup(monkeypatch):
    camera = SimpleNamespace(
        id=uuid4(), rtsp_url="rtsp://admin:secret@camera.local/live", stream_enabled=True,
        name="Camera", resolution_w=1920, resolution_h=1080, fps=25, metadata_={},
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        location=None, roi_polygon=None, calibrations=[],
    )
    key = SimpleNamespace(id=uuid4(), scopes=["camera:read", "camera:write"], camera_id=None)
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.delete = AsyncMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    store = {"session": None}
    def result(statement):
        obj = store["session"] if statement.column_descriptions[0]["entity"] is StreamSession else camera
        return SimpleNamespace(scalar_one_or_none=lambda: obj)
    db.execute.side_effect = result
    db.add.side_effect = lambda session: store.update(session=session)
    service = SimpleNamespace(
        config=Settings(_env_file=None, MEDIAMTX_ALLOWED_RTSP_HOSTS="camera.local"),
        get_path_config=AsyncMock(return_value=desired_path(camera.rtsp_url)),
        runtime_path=AsyncMock(return_value={"online": False, "available": False, "readers": []}),
        signaling=AsyncMock(), check_config=MagicMock(), parse_session_location=MagicMock(),
    )
    async def fake_db():
        yield db
    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[verify_api_key] = lambda: key
    app.dependency_overrides[streams.get_mediamtx] = lambda: service
    monkeypatch.setattr(streams, "reconcile_streams", AsyncMock())
    monkeypatch.setattr(cameras, "reconcile_streams", AsyncMock())
    client = TestClient(app)
    client.headers["X-API-Key"] = "owner-key"
    yield SimpleNamespace(client=client, camera=camera, key=key, db=db, service=service, store=store,
        url=f"/api/v1/cameras/{camera.id}/stream")
    app.dependency_overrides.clear()
    client.close()


@pytest.mark.parametrize(("runtime", "expected"), [
    ({"online": True, "available": True, "tracks2": [{"codec": "H264"}]}, "online"),
    ({"online": True, "available": False, "readers": []}, "idle"),
    ({"online": False, "readers": []}, "idle"),
    ({"online": False, "readers": [{}]}, "offline"),
    (None, "unknown"),
])
def test_stream_status(setup, runtime, expected):
    setup.service.runtime_path.return_value = runtime
    response = setup.client.get(setup.url)
    assert response.status_code == 200
    assert response.json()["status"] == expected
    assert response.json()["whep_url"].endswith(setup.url + "/whep")
    assert "secret" not in response.text


def test_disabled_and_unavailable_status(setup):
    setup.camera.stream_enabled = False
    setup.service.get_path_config.return_value = None
    response = setup.client.get(setup.url)
    assert response.json()["status"] == "disabled"
    assert response.json()["synced"] is True
    assert response.json()["whep_url"] is None
    setup.camera.stream_enabled = True
    setup.service.get_path_config.side_effect = MediaMTXError()
    assert setup.client.get(setup.url).json()["status"] == "unknown"


def test_desired_state_committed_when_mediamtx_down(setup):
    setup.service.get_path_config.side_effect = MediaMTXError()
    response = setup.client.patch(setup.url, json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    setup.db.commit.assert_awaited_once()
    streams.reconcile_streams.assert_awaited_once()


def test_enable_validates_source(setup):
    setup.camera.rtsp_url = "http://invalid-scheme.local/live"
    response = setup.client.patch(setup.url, json={"enabled": True})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_rtsp_source"
    setup.db.commit.assert_not_awaited()


@pytest.mark.parametrize("method", ["get", "patch", "options", "post"])
def test_scope_and_camera_restrictions(setup, method):
    url = setup.url if method in {"get", "patch"} else setup.url + "/whep"
    kwargs = {"json": {"enabled": True}} if method == "patch" else {}
    setup.key.scopes = []
    assert getattr(setup.client, method)(url, **kwargs).status_code == 403
    setup.key.scopes = ["*"]
    setup.key.camera_id = uuid4()
    assert getattr(setup.client, method)(url, **kwargs).status_code == 403
    setup.service.signaling.assert_not_awaited()


def test_missing_key_and_uuid_validation(setup):
    app.dependency_overrides.pop(verify_api_key)
    setup.client.headers.pop("X-API-Key")
    assert setup.client.get(setup.url).status_code == 422
    setup.client.headers["X-API-Key"] = "invalid-key"
    setup.db.execute.side_effect = lambda s: SimpleNamespace(scalar_one_or_none=lambda: None)
    assert setup.client.get(setup.url).status_code == 401


def test_missing_camera_and_invalid_uuid(setup):
    setup.db.execute.side_effect = lambda s: SimpleNamespace(scalar_one_or_none=lambda: None)
    assert setup.client.get(setup.url).status_code == 404
    assert setup.client.get("/api/v1/cameras/not-a-uuid/stream").status_code == 422


def test_whep_options_headers(setup):
    setup.service.signaling.return_value = httpx.Response(204, headers={"Link": '<stun:stun.example.com>; rel="ice-server"', "Accept-Post": "application/sdp"})
    response = setup.client.options(setup.url + "/whep")
    assert response.status_code == 204
    assert response.headers["accept-post"] == "application/sdp"
    assert "stun.example.com" in response.headers["link"]


def offer(setup):
    upstream_id = uuid4()
    setup.service.parse_session_location.return_value = upstream_id
    setup.service.signaling.return_value = httpx.Response(201, content=b"answer", headers={
        "Content-Type": "application/sdp", "Location": f"/{stream_path(setup.camera.id)}/whep/{upstream_id}",
        "ETag": "*", "Link": '<stun:stun.example.com>; rel="ice-server"',
    })
    return setup.client.post(setup.url + "/whep", content="offer", headers={"Content-Type": "application/sdp"})


def test_offer_session_patch_delete(setup):
    response = offer(setup)
    assert response.status_code == 201
    assert response.text == "answer"
    assert response.headers["etag"] == "*"
    location = response.headers["location"]
    session = setup.store["session"]
    assert str(session.upstream_id) not in location
    assert str(session.id) in location
    assert session.owner_hash == hash_key("owner-key")
    setup.service.signaling.return_value = httpx.Response(204)
    response = setup.client.patch(location, content="candidate", headers={"Content-Type": "application/trickle-ice-sdpfrag", "If-Match": "*"})
    assert response.status_code == 204
    assert setup.service.signaling.call_args.kwargs["headers"]["If-Match"] == "*"
    setup.service.signaling.return_value = httpx.Response(404)
    assert setup.client.delete(location).status_code == 204
    setup.db.delete.assert_awaited_once_with(session)


def test_session_ownership_enforced_for_patch_and_delete(setup):
    location = offer(setup).headers["location"]
    setup.client.headers["X-API-Key"] = "another-key"
    setup.service.signaling.reset_mock()
    assert setup.client.delete(location).status_code == 404
    assert setup.client.patch(location, content="candidate").status_code == 404
    setup.service.signaling.assert_not_awaited()


def test_upstream_session_not_found_removes_mapping(setup):
    location = offer(setup).headers["location"]
    setup.service.signaling.side_effect = MediaMTXError("stream_session_not_found", 404)
    response = setup.client.patch(location, content="candidate", headers={"Content-Type": "application/trickle-ice-sdpfrag"})
    assert response.status_code == 404
    setup.db.delete.assert_awaited_once()


def test_offer_db_failure_cleans_up_upstream(setup):
    setup.db.commit.side_effect = RuntimeError("internal path secret")
    response = offer(setup)
    assert response.status_code == 503
    assert "internal" not in response.text
    assert setup.service.signaling.call_args.args[0] == "DELETE"
    setup.db.rollback.assert_awaited_once()


@pytest.mark.parametrize(("body", "content_type", "status"), [
    ("offer", "text/plain", 415), ("", "application/sdp", 400),
    ("a" * 65537, "application/sdp", 413),
])
def test_offer_validation(setup, body, content_type, status):
    assert setup.client.post(setup.url + "/whep", content=body, headers={"Content-Type": content_type}).status_code == status
    setup.service.signaling.assert_not_awaited()


def test_camera_response_redacted_and_source_update_syncs(setup):
    response = setup.client.get(f"/api/v1/cameras/{setup.camera.id}")
    assert response.status_code == 200
    assert response.json()["rtsp_url"] is None
    assert response.json()["rtsp_configured"] is True
    assert "secret" not in response.text
    response = setup.client.patch(f"/api/v1/cameras/{setup.camera.id}", json={"rtsp_url": None})
    assert response.status_code == 200
    assert setup.camera.stream_enabled is False
    cameras.reconcile_streams.assert_awaited_once()


def test_source_change_commits_then_reconciles(setup):
    url = f"/api/v1/cameras/{setup.camera.id}"
    response = setup.client.patch(url, json={"rtsp_url": "rtsp://camera.local/new"})
    assert response.status_code == 200
    assert setup.camera.rtsp_url == "rtsp://camera.local/new"
    assert setup.camera.stream_enabled is True
    setup.db.commit.assert_awaited_once()
    cameras.reconcile_streams.assert_awaited_once()
    setup.db.commit.reset_mock()
    response = setup.client.patch(url, json={"rtsp_url": "http://invalid-scheme.local/live"})
    assert response.status_code == 422
    setup.db.commit.assert_not_awaited()


def test_disabled_stream_cannot_create_or_patch_session(setup):
    location = offer(setup).headers["location"]
    setup.camera.stream_enabled = False
    setup.service.signaling.reset_mock()
    assert setup.client.post(setup.url + "/whep", content="offer", headers={"Content-Type": "application/sdp"}).status_code == 409
    assert setup.client.patch(location, content="candidate", headers={"Content-Type": "application/trickle-ice-sdpfrag"}).status_code == 409
    setup.service.signaling.assert_not_awaited()


def test_ice_restart_answer_preserves_etag_and_content_type(setup):
    location = offer(setup).headers["location"]
    setup.service.signaling.return_value = httpx.Response(200, content=b"restart answer", headers={
        "Content-Type": "application/trickle-ice-sdpfrag", "ETag": '"new-ufrag"',
    })
    response = setup.client.patch(location, content="candidate", headers={"Content-Type": "application/trickle-ice-sdpfrag", "If-Match": "*"})
    assert response.status_code == 200
    assert response.text == "restart answer"
    assert response.headers["etag"] == '"new-ufrag"'
    assert response.headers["content-type"] == "application/trickle-ice-sdpfrag"


def test_camera_delete_commits_before_sync(setup):
    assert setup.client.delete(f"/api/v1/cameras/{setup.camera.id}").status_code == 204
    setup.db.delete.assert_awaited_once_with(setup.camera)
    setup.db.commit.assert_awaited_once()
    cameras.reconcile_streams.assert_awaited_once()


def test_openapi_whep_contract_and_cors_headers(setup):
    schema = setup.client.get("/openapi.json").json()
    path = schema["paths"]["/api/v1/cameras/{camera_id}/stream/whep"]
    assert "application/sdp" in path["post"]["requestBody"]["content"]
    assert "application/sdp" in path["post"]["responses"]["201"]["content"]
    assert path["post"]["security"] == [{"ApiKeyAuth": []}]
    assert "options" in path
    assert {"Location", "ETag", "Link"} <= set(path["post"]["responses"]["201"]["headers"])
    assert "Accept-Post" in path["options"]["responses"]["204"]["headers"]
    patch = schema["paths"]["/api/v1/cameras/{camera_id}/stream/whep/{session_id}"]["patch"]
    assert "application/trickle-ice-sdpfrag" in patch["requestBody"]["content"]
    assert any(parameter["name"] == "If-Match" and parameter["required"] is False for parameter in patch["parameters"])
    assert "ETag" in patch["responses"]["200"]["headers"]
    assert "ETag" in patch["responses"]["204"]["headers"]
    assert "content" not in patch["responses"]["204"]
    middleware = next(m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert set(["Location", "ETag", "Link"]) <= set(middleware.kwargs["expose_headers"])
