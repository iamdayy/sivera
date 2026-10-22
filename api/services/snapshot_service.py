import asyncio
import hashlib
import hmac
import io
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import Settings, settings
from api.core.storage import StorageBackend, generate_storage_key
from api.models.orm import Camera, CameraSnapshot
from api.services.calibration_capture_service import utcnow
from api.services.mediamtx_service import MediaMTXError, validate_rtsp_source
from api.services.rtsp_capture_service import CaptureError, CapturedFrame, capture_frame, check_relay_source, decode_frame
from api.workers.database import DatabaseContext


def source_hash(source: str, config: Settings = settings) -> str:
    return hmac.new(config.SECRET_KEY.encode(), source.encode(), hashlib.sha256).hexdigest()


def require_rtsp_camera(camera: Camera, config: Settings = settings) -> str:
    if not camera.stream_enabled or not camera.rtsp_url:
        raise CaptureError("snapshot_requires_enabled_stream")
    try:
        validate_rtsp_source(camera.rtsp_url, config)
    except MediaMTXError as exc:
        raise CaptureError(exc.code) from None
    return source_hash(camera.rtsp_url, config)


async def latest_snapshot(db: AsyncSession, camera_id: UUID, identity: str, max_age_sec: int) -> CameraSnapshot:
    snapshot = await db.scalar(select(CameraSnapshot).where(
        CameraSnapshot.camera_id == camera_id, CameraSnapshot.source_hash == identity,
    ).order_by(CameraSnapshot.captured_at.desc(), CameraSnapshot.id.desc()).limit(1))
    if snapshot is None:
        raise CaptureError("snapshot_not_found")
    check_freshness(snapshot.captured_at, max_age_sec)
    return snapshot


def check_freshness(captured_at: datetime, max_age_sec: int) -> None:
    age = utcnow() - captured_at
    if age < timedelta(0) or age > timedelta(seconds=max_age_sec):
        raise CaptureError("snapshot_stale")


def read_snapshot(storage: StorageBackend, snapshot: CameraSnapshot, config: Settings = settings) -> CapturedFrame:
    stream = storage.get(snapshot.storage_key)
    if stream is None:
        raise CaptureError("snapshot_image_missing")
    try:
        frame = decode_frame(stream.read(config.CAPTURE_MAX_FRAME_BYTES + 1), config)
    finally:
        stream.close()
    if (frame.width, frame.height) != (snapshot.width, snapshot.height):
        raise CaptureError("snapshot_resolution_mismatch")
    return CapturedFrame(frame.content, frame.width, frame.height, snapshot.captured_at)


async def capture_snapshot(
    camera_id: UUID, db_context: DatabaseContext, storage: StorageBackend,
    config: Settings = settings, expected_hash: str | None = None,
) -> CameraSnapshot:
    async with db_context() as db:
        row = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(Camera.id == camera_id))).one_or_none()
        if row is None:
            raise CaptureError("camera_not_found")
        identity = require_rtsp_camera(row, config)
        source = row.rtsp_url
    if expected_hash is not None and identity != expected_hash:
        raise CaptureError("snapshot_source_changed")
    try:
        await check_relay_source(camera_id, source, config)
    except CaptureError as exc:
        if exc.code == "capture_stream_sync_pending":
            raise CaptureError("snapshot_stream_sync_pending") from None
        raise
    frame = await capture_frame(camera_id, config)
    snapshot = CameraSnapshot(
        id=uuid4(), camera_id=camera_id, source_hash=identity,
        storage_key=generate_storage_key(str(camera_id), output_type="snapshots"),
        width=frame.width, height=frame.height, size_bytes=len(frame.content), captured_at=frame.captured_at,
    )
    discard = True
    try:
        await asyncio.to_thread(storage.save, snapshot.storage_key, io.BytesIO(frame.content), "image/jpeg")
        async with db_context() as db:
            row = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(Camera.id == camera_id).with_for_update())).one_or_none()
            if row is None or require_rtsp_camera(row, config) != identity:
                raise CaptureError("snapshot_source_changed")
            db.add(snapshot)
        discard = False
        return snapshot
    finally:
        if discard:
            await asyncio.to_thread(storage.delete, snapshot.storage_key)
