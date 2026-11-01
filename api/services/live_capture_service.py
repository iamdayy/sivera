from uuid import UUID

from sqlalchemy import select

from api.core.config import Settings, settings
from api.core.database import DatabaseContext, get_db_context
from api.models.orm import Camera
from api.services.rtsp_capture_service import CaptureError, CapturedFrame, capture_frame, check_relay_source
from api.services.snapshot_service import require_rtsp_camera


async def capture_live_frame(
    camera_id: UUID, config: Settings = settings, expected_hash: str | None = None,
    db_context: DatabaseContext = get_db_context,
) -> CapturedFrame:
    async with db_context() as db:
        camera = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(
            Camera.id == camera_id,
        ))).one_or_none()
        if camera is None:
            raise CaptureError("camera_not_found")
        identity = require_rtsp_camera(camera, config)
        if expected_hash is not None and identity != expected_hash:
            raise CaptureError("prediction_source_changed")
        source = camera.rtsp_url
    await check_relay_source(camera_id, source, config)
    return await capture_frame(camera_id, config)
