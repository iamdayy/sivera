import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from api.core.config import Settings
from api.core.database import get_db
from api.main import app
from api.routes import public_stream
from api.routes.streams import get_mediamtx
from api.services.mediamtx_service import MediaMTXService, stream_path


@pytest.fixture
def setup():
    camera = SimpleNamespace(id=uuid4(), stream_enabled=True)
    session_id = uuid4()
    db = MagicMock()
    db.execute = AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: camera))
    requests = []
    replies = {
        "OPTIONS": httpx.Response(204, headers={
            "Accept-Post": "application/sdp", "Accept-Patch": "application/trickle-ice-sdpfrag",
            "Link": '<stun:stun.example.test>; rel="ice-server"',
            "Access-Control-Allow-Origin": "*",
        }),
        "POST": httpx.Response(201, content=b"v=0\r\na=ice-ufrag:remote\r\n", headers={
            "Content-Type": "application/sdp", "ETag": '"first"',
            "Location": f"/{stream_path(camera.id)}/whep/{session_id}",
            "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "Link",
        }),
        "PATCH": httpx.Response(204, headers={"ETag": '"second"'}),
        "DELETE": httpx.Response(200),
    }

    def upstream(request):
        requests.append(request)
        return replies[request.method]

    service = MediaMTXService(Settings(
        _env_file=None, MEDIAMTX_ENABLED=True, MEDIAMTX_READ_PASSWORD="test-reader-password",
    ), httpx.MockTransport(upstream))

    async def database():
        yield db

    test_app = FastAPI()
    test_app.include_router(public_stream.router, prefix="/api/v1/public")
    test_app.add_middleware(CORSMiddleware, allow_origins=["https://dashboard.example.test"],
        allow_methods=["*"], allow_headers=["*"],
        expose_headers=["Location", "ETag", "Link", "Accept-Post", "Accept-Patch"])
    test_app.dependency_overrides[get_db] = database
    test_app.dependency_overrides[get_mediamtx] = lambda: service
    with TestClient(test_app) as client:
        client.headers["Origin"] = "https://dashboard.example.test"
        yield SimpleNamespace(client=client, camera=camera, db=db, service=service,
            requests=requests, replies=replies, session_id=session_id,
            url=f"/api/v1/public/cameras/{camera.id}/stream/whep")
    asyncio.run(service.close())


def offer(setup):
    return setup.client.post(setup.url, content="v=0\r\na=ice-ufrag:local\r\n",
        headers={"Content-Type": "application/sdp"})


def test_public_options_is_204_with_real_capabilities(setup):
    response = setup.client.options(setup.url)
    assert response.status_code == 204
    assert response.content == b""
    assert response.headers["accept-post"] == "application/sdp"
    assert response.headers["accept-patch"] == "application/trickle-ice-sdpfrag"
    assert response.headers["link"] == '<stun:stun.example.test>; rel="ice-server"'
    assert response.headers["cache-control"] == "no-store"
    assert len(setup.requests) == 1
    assert setup.requests[0].headers["authorization"].startswith("Basic ")
    assert "X-API-Key" not in setup.client.headers


def test_cors_preflight_is_separate_from_whep_options(setup):
    response = setup.client.options(setup.url, headers={
        "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://dashboard.example.test"
    assert setup.requests == []
    setup.db.execute.assert_not_awaited()


@pytest.mark.parametrize("absolute_location", [False, True])
def test_offer_rewrites_location_and_preserves_cors_visible_headers(setup, absolute_location):
    if absolute_location:
        setup.replies["POST"].headers["Location"] = setup.service.whep_base + setup.replies["POST"].headers["Location"]
    response = offer(setup)
    assert response.status_code == 201
    assert response.content == setup.replies["POST"].content
    assert response.headers["content-type"] == "application/sdp"
    assert response.headers["location"] == f"http://testserver{setup.url}/{setup.session_id}"
    assert response.headers["etag"] == '"first"'
    assert response.headers["access-control-allow-origin"] == "https://dashboard.example.test"
    exposed = set(response.headers["access-control-expose-headers"].lower().replace(" ", "").split(","))
    assert {"location", "etag", "link", "accept-post", "accept-patch"} <= exposed
    assert len(setup.requests) == 1
    assert setup.requests[0].url.path == f"/{stream_path(setup.camera.id)}/whep"
    assert setup.requests[0].content == b"v=0\r\na=ice-ufrag:local\r\n"
    assert "test-reader-password" not in str(response.headers)


def test_upstream_cors_headers_cannot_override_backend_policy(setup):
    setup.client.headers["Origin"] = "https://other.example.test"
    for response in [setup.client.options(setup.url), offer(setup)]:
        assert "access-control-allow-origin" not in response.headers


def test_public_patch_and_delete_use_returned_session_location(setup):
    location = offer(setup).headers["location"]
    response = setup.client.patch(location, content="a=ice-ufrag:local\r\n", headers={
        "Content-Type": "application/trickle-ice-sdpfrag", "If-Match": '"first"',
    })
    assert response.status_code == 204
    assert response.headers["etag"] == '"second"'
    assert setup.requests[-1].headers["if-match"] == '"first"'
    assert setup.requests[-1].url.path == f"/{stream_path(setup.camera.id)}/whep/{setup.session_id}"
    response = setup.client.delete(location)
    assert response.status_code == 204
    assert response.content == b""
    assert setup.requests[-1].method == "DELETE"


def test_public_ice_restart_preserves_fragment_and_etag(setup):
    setup.replies["PATCH"] = httpx.Response(200, content=b"a=ice-ufrag:restart\r\n", headers={
        "Content-Type": "application/trickle-ice-sdpfrag", "ETag": '"restart"',
    })
    response = setup.client.patch(f"{setup.url}/{setup.session_id}", content="a=ice-ufrag:local\r\n",
        headers={"Content-Type": "application/trickle-ice-sdpfrag"})
    assert response.status_code == 200
    assert response.content == setup.replies["PATCH"].content
    assert response.headers["content-type"] == "application/trickle-ice-sdpfrag"
    assert response.headers["etag"] == '"restart"'


@pytest.mark.parametrize("status", [200, 204, 404])
def test_public_delete_normalizes_mediamtx_cleanup_status(setup, status):
    setup.replies["DELETE"] = httpx.Response(status)
    response = setup.client.delete(f"{setup.url}/{setup.session_id}")
    assert response.status_code == 204
    assert response.content == b""


@pytest.mark.parametrize("location", ["", "https://external.example.test/whep/session", "/wrong/whep/session"])
def test_invalid_upstream_location_is_not_forwarded(setup, location):
    setup.replies["POST"].headers["Location"] = location
    response = offer(setup)
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_stream_response"
    assert "location" not in response.headers
    assert len(setup.requests) == 1


@pytest.mark.parametrize(("body", "mime", "status"), [
    ("offer", "text/plain", 415), ("", "application/sdp", 400), ("a" * 65537, "application/sdp", 413),
])
def test_public_offer_validation(setup, body, mime, status):
    response = setup.client.post(setup.url, content=body, headers={"Content-Type": mime})
    assert response.status_code == status
    assert setup.requests == []


def test_unavailable_mediamtx_returns_safe_error_without_retry(setup):
    setup.replies["POST"] = httpx.Response(503, text="rtsp://private:credential@internal/live")
    response = offer(setup)
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "stream_unavailable"}}
    assert len(setup.requests) == 1


def test_disabled_missing_camera_and_invalid_session(setup):
    setup.camera.stream_enabled = False
    assert setup.client.options(setup.url).status_code == 403
    assert offer(setup).status_code == 403
    setup.db.execute.return_value = SimpleNamespace(scalar_one_or_none=lambda: None)
    assert setup.client.options(setup.url).status_code == 404
    assert setup.client.delete(setup.url + "/not-a-uuid").status_code == 422
    assert setup.requests == []


def test_public_openapi_matches_signaling_contract():
    paths = app.openapi()["paths"]
    path = paths["/api/v1/public/cameras/{camera_id}/stream/whep"]
    assert path["options"]["security"] == []
    assert "204" in path["options"]["responses"]
    assert "Accept-Post" in path["options"]["responses"]["204"]["headers"]
    assert path["post"]["security"] == []
    assert "application/sdp" in path["post"]["requestBody"]["content"]
    assert "application/sdp" in path["post"]["responses"]["201"]["content"]
    assert {"Location", "ETag", "Link"} <= set(path["post"]["responses"]["201"]["headers"])
    session = paths["/api/v1/public/cameras/{camera_id}/stream/whep/{session_id}"]
    assert "application/trickle-ice-sdpfrag" in session["patch"]["requestBody"]["content"]
    assert session["patch"]["security"] == session["delete"]["security"] == []
    assert "204" in session["delete"]["responses"]
