import asyncio
import io
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from api.core.config import Settings
from api.core.storage import LocalStorage
from api.models.orm import CameraSnapshot
from api.services import snapshot_service as service
from api.services.calibration_capture_service import utcnow
from api.services.mediamtx_service import desired_path
from api.services.rtsp_capture_service import CaptureError, decode_frame
from tests.test_calibration_capture_service import jpeg


def test_identity_no_plain_credentials_and_freshness():
    config = Settings(_env_file=None, SECRET_KEY="test-secret")
    assert len(service.source_hash("rtsp://private:password@camera/live", config)) == 64
    assert service.source_hash("rtsp://camera/a", config) != service.source_hash("rtsp://camera/b", config)
    service.check_freshness(utcnow() - timedelta(seconds=1), 5)
    for time in (utcnow() - timedelta(seconds=31), utcnow() + timedelta(seconds=10)):
        with pytest.raises(CaptureError, match="snapshot_stale"):
            service.check_freshness(time, 30)


def test_snapshot_original_dimensions_and_timestamp(tmp_path):
    storage = LocalStorage(tmp_path)
    snapshot = CameraSnapshot(storage_key="frame.jpg", width=640, height=480, captured_at=utcnow() - timedelta(seconds=5))
    storage.save(snapshot.storage_key, io.BytesIO(jpeg()))
    frame = service.read_snapshot(storage, snapshot)
    assert frame.captured_at == snapshot.captured_at
    assert (frame.width, frame.height) == (640, 480)
    snapshot.width = 320
    with pytest.raises(CaptureError, match="snapshot_resolution_mismatch"):
        service.read_snapshot(storage, snapshot)


def test_capture_source_changed_discards_uncommitted_snapshot(monkeypatch, tmp_path):
    row = SimpleNamespace(rtsp_url="rtsp://camera.local/a", stream_enabled=True)
    db = MagicMock(execute=AsyncMock(return_value=SimpleNamespace(one_or_none=lambda: row)))
    @asynccontextmanager
    async def context():
        yield db
    monkeypatch.setattr(service, "check_relay_source", AsyncMock())
    async def capture(*args):
        row.rtsp_url = "rtsp://camera.local/b"
        return decode_frame(jpeg())
    monkeypatch.setattr(service, "capture_frame", capture)
    storage = LocalStorage(tmp_path)
    with pytest.raises(CaptureError, match="snapshot_source_changed"):
        asyncio.run(service.capture_snapshot(uuid4(), context, storage, Settings(_env_file=None, MEDIAMTX_ALLOWED_RTSP_HOSTS="camera.local")))
    db.add.assert_not_called()
    assert list(tmp_path.rglob("*.jpg")) == []
