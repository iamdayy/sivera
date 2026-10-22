"""Opt-in checks against disposable PostgreSQL and MediaMTX instances."""
import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.core.config import Settings, settings
from api.services.mediamtx_service import MediaMTXError, MediaMTXService, stream_path


def test_camera_stream_migration(monkeypatch):
    url = os.environ.get("SIVERA_TEST_DATABASE_URL")
    if not url:
        pytest.skip("SIVERA_TEST_DATABASE_URL must point to an empty disposable database")
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    cfg = Config(str(Path(__file__).resolve().parents[1] / "api/alembic.ini"))
    command.upgrade(cfg, "002_add_is_active_to_depth_scales")
    camera_id = uuid4()
    async def inspect_schema(stage):
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                if stage == "before":
                    await conn.execute(text("INSERT INTO cameras (id,name,resolution_w,resolution_h,fps,metadata) VALUES (:id,'fixture',640,480,25,'{}')"), {"id": camera_id})
                elif stage == "after":
                    assert await conn.scalar(text("SELECT stream_enabled FROM cameras WHERE id=:id"), {"id": camera_id}) is False
                    await conn.execute(text("INSERT INTO stream_sessions (id,camera_id,owner_hash,upstream_id,expires_at) VALUES (:id,:camera,:owner,:upstream,:expires)"), {
                        "id": uuid4(), "camera": camera_id, "owner": "a" * 64,
                        "upstream": uuid4(), "expires": datetime.now(timezone.utc),
                    })
                    await conn.execute(text("DELETE FROM cameras WHERE id=:id"), {"id": camera_id})
                    assert await conn.scalar(text("SELECT count(*) FROM stream_sessions")) == 0
                elif stage == "downgraded":
                    assert await conn.scalar(text("SELECT to_regclass('public.stream_sessions')")) is None
        finally:
            await engine.dispose()
    asyncio.run(inspect_schema("before"))
    command.upgrade(cfg, "head")
    asyncio.run(inspect_schema("after"))
    command.downgrade(cfg, "002_add_is_active_to_depth_scales")
    asyncio.run(inspect_schema("downgraded"))
    command.upgrade(cfg, "head")


def test_real_mediamtx_control_and_whep():
    port = os.environ.get("SIVERA_TEST_MEDIAMTX_API_PORT")
    if not port:
        pytest.skip("SIVERA_TEST_MEDIAMTX_API_PORT must point to disposable MediaMTX v1.20.0")
    config = Settings(
        _env_file=None, MEDIAMTX_ENABLED=True, MEDIAMTX_HOST="127.0.0.1",
        MEDIAMTX_API_PORT=int(port), MEDIAMTX_WEBRTC_PORT=int(os.environ["SIVERA_TEST_MEDIAMTX_WHEP_PORT"]),
        MEDIAMTX_API_PASSWORD=os.environ["SIVERA_TEST_MEDIAMTX_API_PASSWORD"],
        MEDIAMTX_READ_PASSWORD=os.environ["SIVERA_TEST_MEDIAMTX_READ_PASSWORD"],
        MEDIAMTX_ALLOWED_RTSP_HOSTS="192.0.2.1",
    )
    async def check():
        camera_id = uuid4()
        name = stream_path(camera_id)
        service = MediaMTXService(config)
        try:
            async with httpx.AsyncClient() as client:
                assert (await client.get(service.api_url + "/v3/paths/list")).status_code == 401
                assert (await client.get(service.api_url + "/v3/paths/list", auth=(config.MEDIAMTX_API_USER, "wrong"))).status_code == 401
            await service.ensure_path(name, "rtsp://192.0.2.1/live", None)
            current = await service.get_path_config(name)
            assert current["sourceOnDemand"] is True
            assert current["rtspTransport"] == "tcp"
            await service.ensure_path(name, "rtsp://192.0.2.1/live", current)
            response = await service.signaling("OPTIONS", camera_id)
            assert response.status_code == 204
            assert response.headers["accept-post"] == "application/sdp"
            async with httpx.AsyncClient() as client:
                assert (await client.options(service.whep_base + f"/{name}/whep")).status_code == 401
            with pytest.raises(MediaMTXError):
                await service.signaling("POST", camera_id, content=b"invalid-sdp", headers={"Content-Type": "application/sdp"})
            await service.delete_path(name)
            await service.delete_path(name)
            assert await service.get_path_config(name) is None
        finally:
            await service.close()
    asyncio.run(check())


def test_real_backend_stream_lifecycle(monkeypatch):
    url = os.environ.get("SIVERA_TEST_DATABASE_URL")
    port = os.environ.get("SIVERA_TEST_MEDIAMTX_API_PORT")
    if not url or not port:
        pytest.skip("Requires both disposable database and MediaMTX environment variables")
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from api.main import app
    from api.core.database import get_db
    from api.core.security import verify_api_key
    from api.routes.streams import get_mediamtx
    from api.services import stream_service
    config = Settings(
        _env_file=None, MEDIAMTX_ENABLED=True, MEDIAMTX_HOST="127.0.0.1",
        MEDIAMTX_API_PORT=int(port), MEDIAMTX_WEBRTC_PORT=int(os.environ["SIVERA_TEST_MEDIAMTX_WHEP_PORT"]),
        MEDIAMTX_API_PASSWORD=os.environ["SIVERA_TEST_MEDIAMTX_API_PASSWORD"],
        MEDIAMTX_READ_PASSWORD=os.environ["SIVERA_TEST_MEDIAMTX_READ_PASSWORD"],
        MEDIAMTX_ALLOWED_RTSP_HOSTS="192.0.2.1,127.0.0.1",
    )
    async def check():
        engine = create_async_engine(url)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        @asynccontextmanager
        async def context():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
        async def dependency():
            async with context() as session:
                yield session
        service = MediaMTXService(config)
        monkeypatch.setattr(stream_service, "get_db_context", context)
        app.dependency_overrides[get_db] = dependency
        app.dependency_overrides[verify_api_key] = lambda: SimpleNamespace(scopes=["*"], camera_id=None)
        app.dependency_overrides[get_mediamtx] = lambda: service
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", headers={"X-API-Key": "integration-key"}) as client:
                source = os.environ.get("SIVERA_TEST_RTSP_SOURCE", "rtsp://192.0.2.1/live")
                response = await client.post("/api/v1/cameras", json={"name": "fixture", "resolution_w": 640, "resolution_h": 480, "rtsp_url": source})
                assert response.status_code == 201, response.text
                camera_id = response.json()["id"]
                assert response.json()["rtsp_url"] is None
                assert response.json()["stream_enabled"] is False
                path = f"/api/v1/cameras/{camera_id}/stream"
                assert (await client.get("/health")).json()["database"] == "connected"
                response = await client.get("/openapi.json")
                assert response.status_code == 200
                published = json.loads((Path(__file__).resolve().parents[1] / "openapi.json").read_text())
                def without_defaults(value):
                    if isinstance(value, dict):
                        return {key: without_defaults(item) for key, item in value.items() if key != "default"}
                    if isinstance(value, list):
                        return [without_defaults(item) for item in value]
                    return value
                # Deployment settings may change defaults, but not the request/response/security contract.
                for section in ("paths", "components"):
                    assert without_defaults(response.json()[section]) == without_defaults(published[section])
                assert (await client.patch(path, json={"enabled": True})).status_code == 200
                response = await client.get(path)
                assert response.status_code == 200, response.text
                assert response.json()["synced"] is True
                assert response.json()["status"] == "idle"
                assert (await client.options(path + "/whep")).status_code == 204
                if "SIVERA_TEST_RTSP_SOURCE" in os.environ:
                    fingerprint = ":".join(["00"] * 32)
                    offer = "\r\n".join([
                        "v=0", "o=- 0 0 IN IP4 127.0.0.1", "s=-", "t=0 0",
                        "a=group:BUNDLE 0", "a=ice-options:trickle",
                        "m=video 9 UDP/TLS/RTP/SAVPF 96", "c=IN IP4 0.0.0.0",
                        "a=mid:0", "a=recvonly", "a=rtcp-mux", "a=setup:actpass",
                        "a=ice-ufrag:siveratest", "a=ice-pwd:siveraTestPassword12345678",
                        "a=fingerprint:sha-256 " + fingerprint,
                        "a=rtpmap:96 H264/90000",
                        "a=fmtp:96 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f",
                        "",
                    ])
                    response = await client.post(path + "/whep", content=offer, headers={"Content-Type": "application/sdp"})
                    assert response.status_code == 201, response.text
                    assert response.headers["content-type"] == "application/sdp"
                    assert "a=candidate:" in response.text
                    assert response.headers["location"].startswith("http://test" + path)
                    session_url = response.headers["location"]
                    fragment = "\r\n".join([
                        "a=ice-ufrag:siveratest", "a=ice-pwd:siveraTestPassword12345678",
                        "m=video 9 UDP/TLS/RTP/SAVPF 96", "a=mid:0",
                        "a=candidate:1 1 UDP 2130706431 127.0.0.1 40000 typ host", "",
                    ])
                    response = await client.patch(session_url, content=fragment, headers={"Content-Type": "application/trickle-ice-sdpfrag", "If-Match": "*"})
                    assert response.status_code == 204, response.text
                    assert (await client.delete(session_url)).status_code == 204
                    assert (await client.delete(session_url)).status_code == 404
                if "SIVERA_TEST_MEDIAMTX_CONTAINER" in os.environ:
                    container = os.environ["SIVERA_TEST_MEDIAMTX_CONTAINER"]
                    subprocess.run(["docker", "restart", container], check=True, capture_output=True)
                    # Docker may reassign ephemeral host ports after restart.
                    mapped = subprocess.run(["docker", "port", container, "9997/tcp"], check=True, capture_output=True, text=True).stdout.strip()
                    service.api_url = "http://" + mapped
                    for _ in range(30):
                        try:
                            await service.list_path_configs()
                            break
                        except MediaMTXError:
                            await asyncio.sleep(1)
                    else:
                        pytest.fail("Disposable MediaMTX did not recover after restart")
                    assert await service.get_path_config(stream_path(camera_id)) is None
                    await stream_service.reconcile_streams(service)
                    assert await service.get_path_config(stream_path(camera_id)) is not None
                assert (await client.patch(path, json={"enabled": False})).status_code == 200
                assert await service.get_path_config(stream_path(camera_id)) is None
                assert (await client.delete(f"/api/v1/cameras/{camera_id}")).status_code == 204
        finally:
            app.dependency_overrides.clear()
            await service.close()
            await engine.dispose()
    asyncio.run(check())
