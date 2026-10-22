from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from api.services import stream_service
from api.services.mediamtx_service import MediaMTXError, stream_path


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def setup(monkeypatch):
    enabled = SimpleNamespace(id=uuid4(), stream_enabled=True, rtsp_url="rtsp://camera.local/live")
    disabled = SimpleNamespace(id=uuid4(), stream_enabled=False, rtsp_url="rtsp://camera.local/other")
    orphan = stream_path(uuid4())
    expired = SimpleNamespace(camera_id=enabled.id, upstream_id=uuid4())
    db = MagicMock()
    db.scalar = AsyncMock(return_value=True)
    db.execute = AsyncMock(side_effect=[
        SimpleNamespace(all=lambda: [enabled, disabled]),
        SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [expired])),
    ])
    db.delete = AsyncMock()
    @asynccontextmanager
    async def context():
        yield db
    monkeypatch.setattr(stream_service, "get_db_context", context)
    service = SimpleNamespace(
        config=SimpleNamespace(MEDIAMTX_ENABLED=True),
        list_path_configs=AsyncMock(return_value=[
            {"name": orphan}, {"name": stream_path(disabled.id)},
            {"name": "external_stream"}, {"name": "sivera_custom"},
        ]),
        ensure_path=AsyncMock(), delete_path=AsyncMock(), signaling=AsyncMock(),
    )
    return SimpleNamespace(service=service, db=db, enabled=enabled, disabled=disabled, orphan=orphan, expired=expired)


@pytest.mark.anyio
async def test_reconcile_only_uuid_namespace_and_cleanup(setup):
    await stream_service.reconcile_streams(setup.service)
    setup.service.ensure_path.assert_awaited_once_with(stream_path(setup.enabled.id), setup.enabled.rtsp_url, None)
    deleted = {call.args[0] for call in setup.service.delete_path.await_args_list}
    assert deleted == {setup.orphan, stream_path(setup.disabled.id)}
    setup.service.signaling.assert_awaited_once_with("DELETE", setup.expired.camera_id, setup.expired.upstream_id)
    setup.db.delete.assert_awaited_once_with(setup.expired)


@pytest.mark.anyio
async def test_reconcile_skips_when_other_process_holds_lock(setup):
    setup.db.scalar.return_value = False
    await stream_service.reconcile_streams(setup.service)
    setup.service.list_path_configs.assert_not_awaited()


@pytest.mark.anyio
async def test_reconcile_recovers_after_outage(setup):
    setup.service.list_path_configs.side_effect = [MediaMTXError(), []]
    await stream_service.reconcile_streams(setup.service)
    setup.service.ensure_path.assert_not_awaited()
    setup.db.execute.side_effect = [
        SimpleNamespace(all=lambda: [setup.enabled, setup.disabled]),
        SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [])),
    ]
    await stream_service.reconcile_streams(setup.service)
    setup.service.ensure_path.assert_awaited_once()


@pytest.mark.anyio
async def test_removed_allowlist_source_is_deleted(setup):
    name = stream_path(setup.enabled.id)
    setup.service.list_path_configs.return_value.append({"name": name})
    setup.service.ensure_path.side_effect = MediaMTXError("rtsp_host_not_allowed", 422)
    await stream_service.reconcile_streams(setup.service)
    assert name in {call.args[0] for call in setup.service.delete_path.await_args_list}
