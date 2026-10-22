import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import or_, select, text

from api.core.database import get_db_context
from api.models.orm import Camera, StreamSession
from api.services.mediamtx_service import MANAGED_PATH, MediaMTXError, MediaMTXService, stream_path


logger = logging.getLogger(__name__)


async def reconcile_streams(service: MediaMTXService) -> None:
    if not service.config.MEDIAMTX_ENABLED:
        return
    try:
        async with get_db_context() as db:
            # One reconciler across API processes; all source reads happen after the lock.
            acquired = await db.scalar(text("SELECT pg_try_advisory_xact_lock(736483120)"))
            if not acquired:
                return
            rows = (await db.execute(select(Camera.id, Camera.rtsp_url, Camera.stream_enabled))).all()
            desired = {stream_path(row.id): row for row in rows if row.stream_enabled and row.rtsp_url}
            configs = {item["name"]: item for item in await service.list_path_configs()}
            for name, row in desired.items():
                try:
                    await service.ensure_path(name, row.rtsp_url, configs.get(name))
                except MediaMTXError as exc:
                    logger.warning("Stream sync camera=%s code=%s", row.id, exc.code)
                    # Invalidated allowlist entries must not remain readable.
                    if exc.status_code == 422 and name in configs:
                        await service.delete_path(name)
            for name in configs.keys() - desired.keys():
                if MANAGED_PATH.fullmatch(name):
                    await service.delete_path(name)
            enabled_ids = [row.id for row in desired.values()]
            expired = (await db.execute(select(StreamSession).where(or_(
                StreamSession.expires_at <= datetime.now(timezone.utc),
                StreamSession.camera_id.not_in(enabled_ids),
            )))).scalars().all()
            for session in expired:
                await service.signaling("DELETE", session.camera_id, session.upstream_id)
                await db.delete(session)
    except MediaMTXError as exc:
        logger.warning("Stream reconciliation deferred code=%s", exc.code)
    except Exception:
        # Do not log DB exception strings: they may include source URLs/parameters.
        logger.error("Stream reconciliation failed; check database and MediaMTX configuration")


async def reconciliation_loop(service: MediaMTXService) -> None:
    while True:
        await reconcile_streams(service)
        await asyncio.sleep(service.config.MEDIAMTX_RECONCILE_INTERVAL)
