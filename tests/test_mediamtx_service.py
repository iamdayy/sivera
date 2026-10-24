import json
import logging
from uuid import uuid4

import httpx
import pytest

from api.core.config import Settings
from api.services.mediamtx_service import (
    MediaMTXError, MediaMTXService, SignalingLogFilter, desired_path,
    stream_path, validate_rtsp_source,
)


@pytest.fixture
def config():
    return Settings(
        _env_file=None, MEDIAMTX_ENABLED=True,
        MEDIAMTX_API_PASSWORD="control-secret", MEDIAMTX_READ_PASSWORD="read-secret",
        MEDIAMTX_ALLOWED_RTSP_HOSTS="camera.local,192.168.1.10",
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("source", [
    "http://camera.local/live", "rtsp://camera.local:0/live",
    "rtsp://camera.local:bad/live", "rtsp://camera.local/live#fragment",
    "rtsp://camera.local/live\n", "rtsp://camera.local/$MTX_QUERY", "rtsp:///live",
    "rtsp://user@camera.local/live", "rtsp://:password@camera.local/live",
    "rtsp://camera.local/%bad%", "RTSP://camera.local/live", "rtsp://camera.local/\\live",
])
def test_source_rejected_without_leaking_credentials(source, config):
    with pytest.raises(MediaMTXError) as exc:
        validate_rtsp_source(source, config)
    assert exc.value.status_code == 422
    assert source not in str(exc.value)


@pytest.mark.parametrize("source", ["rtsp://u:p@camera.local/live", "rtsps://192.168.1.10:322/live", "rtsp://evil.local/live"])
def test_valid_rtsp_sources(source, config):
    validate_rtsp_source(source, config)


@pytest.mark.anyio
async def test_idempotent_path_creation_and_patch(config):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"status": "ok"})
    service = MediaMTXService(config, httpx.MockTransport(handler))
    name = stream_path(uuid4())
    source = "rtsp://camera.local/live"
    try:
        await service.ensure_path(name, source, None)
        await service.ensure_path(name, source, desired_path(source))
        await service.ensure_path(name, source, desired_path("rtsp://camera.local/old"))
        assert [r.method for r in calls] == ["POST", "PATCH"]
        assert calls[0].headers["authorization"].startswith("Basic ")
        assert json.loads(calls[0].content)["rtspTransport"] == "tcp"
        assert json.loads(calls[0].content)["sourceOnDemand"] is True
    finally:
        await service.close()


@pytest.mark.anyio
async def test_control_pagination_and_not_found(config):
    def handler(request):
        if "/get/" in request.url.path:
            return httpx.Response(404)
        page = int(request.url.params["page"])
        return httpx.Response(200, json={"pageCount": 2, "items": [{"name": str(page)}]})
    service = MediaMTXService(config, httpx.MockTransport(handler))
    try:
        assert await service.get_path_config("missing") is None
        assert await service.runtime_path("missing") is None
        assert await service.list_path_configs() == [{"name": "0"}, {"name": "1"}]
    finally:
        await service.close()


@pytest.mark.anyio
async def test_get_retry_but_sdp_post_is_never_retried(config):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("secret internal host", request=request)
    service = MediaMTXService(config, httpx.MockTransport(handler))
    try:
        with pytest.raises(MediaMTXError, match="^stream_unavailable$"):
            await service.get_path_config("camera")
        assert len(calls) == 3
        calls.clear()
        with pytest.raises(MediaMTXError):
            await service.signaling("POST", uuid4(), content=b"offer")
        assert len(calls) == 1
    finally:
        await service.close()


@pytest.mark.parametrize("status", [301, 401, 403, 500])
@pytest.mark.anyio
async def test_upstream_errors_are_sanitized(status, config):
    service = MediaMTXService(config, httpx.MockTransport(lambda r: httpx.Response(status, text="rtsp://admin:secret@private/live")))
    try:
        with pytest.raises(MediaMTXError, match="^stream_unavailable$"):
            await service.signaling("POST", uuid4(), content=b"offer")
    finally:
        await service.close()


@pytest.mark.anyio
async def test_response_size_bound(config):
    service = MediaMTXService(config, httpx.MockTransport(lambda r: httpx.Response(201, content=b"a" * 65537)))
    try:
        with pytest.raises(MediaMTXError, match="invalid_stream_response"):
            await service.signaling("POST", uuid4(), content=b"offer")
    finally:
        await service.close()


def test_location_validation_and_log_redaction(config):
    service = MediaMTXService(config)
    camera_id, session_id = uuid4(), uuid4()
    path = f"/{stream_path(camera_id)}/whep/{session_id}"
    assert service.parse_session_location(path, camera_id) == session_id
    assert service.parse_session_location(service.whep_base + path, camera_id) == session_id
    for location in ["https://evil.local" + path, path + "?token=secret", path + "/extra", f"/{stream_path(uuid4())}/whep/{session_id}"]:
        with pytest.raises(MediaMTXError):
            service.parse_session_location(location, camera_id)
    record = logging.LogRecord("httpx", logging.INFO, "", 0, "HTTP %s", (service.whep_base + path,), None)
    assert SignalingLogFilter().filter(record)
    assert str(session_id) not in record.getMessage()


@pytest.mark.anyio
async def test_unconfigured_service_fails_closed():
    service = MediaMTXService(Settings(_env_file=None))
    try:
        with pytest.raises(MediaMTXError, match="stream_service_not_configured"):
            await service.get_path_config("camera")
    finally:
        await service.close()
