import asyncio
import io
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import Settings, settings
from api.core.database import get_db_context
from api.core.storage import StorageBackend, generate_storage_key
from api.models.orm import Camera, CalibrationCaptureImage, CalibrationCaptureSession
from api.services.mediamtx_service import MediaMTXError, validate_rtsp_source
from api.services.rtsp_capture_service import CaptureError, CapturedFrame, capture_frame


logger = logging.getLogger(__name__)
ACTIVE_STATUSES = ("queued", "running", "stopping")
DISPATCH_INTERVAL_SEC = 15
DatabaseContext = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def finish(session: CalibrationCaptureSession, status: str, reason: str, now: datetime) -> None:
    session.status = status
    session.finish_reason = reason
    session.completed_at = now
    session.worker_token = None


def expire_session(session: CalibrationCaptureSession, now: datetime, config: Settings = settings) -> bool:
    if session.status not in ACTIVE_STATUSES:
        return False
    if session.status == "stopping":
        finish(session, "stopped", "user_stop", now)
    elif now >= session.deadline_at:
        finish(session, "expired", "deadline", now)
    elif session.status == "queued" and now >= session.created_at + timedelta(seconds=config.CAPTURE_QUEUE_TIMEOUT_SEC):
        finish(session, "failed", "queue_timeout", now)
        session.error_code = "capture_queue_timeout"
    elif session.status == "running" and (
        session.heartbeat_at is None or now >= session.heartbeat_at + timedelta(seconds=config.CAPTURE_HEARTBEAT_TIMEOUT_SEC)
    ):
        finish(session, "failed", "worker_lost", now)
        session.error_code = "capture_worker_lost"
    else:
        return False
    return True


async def publish_capture(session_id: UUID) -> None:
    from api.workers.tasks.capture_task import run_capture_task

    await asyncio.to_thread(run_capture_task.apply_async, args=[str(session_id)], queue="capture", retry=False)


async def dispatch_capture(db: AsyncSession, session_id: UUID) -> None:
    now = utcnow()
    try:
        dispatched = await db.scalar(update(CalibrationCaptureSession).where(
            CalibrationCaptureSession.id == session_id,
            CalibrationCaptureSession.status == "queued",
            (CalibrationCaptureSession.last_dispatched_at.is_(None)) |
            (CalibrationCaptureSession.last_dispatched_at <= now - timedelta(seconds=DISPATCH_INTERVAL_SEC)),
        ).values(last_dispatched_at=now).returning(CalibrationCaptureSession.id))
        await db.commit()
    except Exception:
        await db.rollback()
        logger.warning("Capture dispatch deferred session=%s", session_id)
        return
    if dispatched:
        try:
            await publish_capture(dispatched)
        except Exception:
            # The committed row is an outbox; the API reconciler retries dispatch.
            logger.warning("Capture dispatch deferred session=%s", session_id)


async def reconcile_captures(config: Settings = settings) -> None:
    try:
        dispatch = []
        async with get_db_context() as db:
            now = utcnow()
            rows = (await db.execute(select(CalibrationCaptureSession).where(
                CalibrationCaptureSession.status.in_(ACTIVE_STATUSES),
            ).with_for_update(skip_locked=True))).scalars().all()
            for session in rows:
                expire_session(session, now, config)
                if session.status == "queued" and (
                    session.last_dispatched_at is None or
                    session.last_dispatched_at <= now - timedelta(seconds=DISPATCH_INTERVAL_SEC)
                ):
                    session.last_dispatched_at = now
                    dispatch.append(session.id)
        for session_id in dispatch:
            try:
                await publish_capture(session_id)
            except Exception:
                logger.warning("Capture dispatch deferred session=%s", session_id)
    except Exception:
        logger.error("Capture reconciliation failed; check database and queue configuration")


async def capture_reconciliation_loop(config: Settings = settings) -> None:
    while True:
        await reconcile_captures(config)
        await asyncio.sleep(config.CAPTURE_RECONCILE_INTERVAL_SEC)


class CaptureRunner:
    def __init__(self, session_id: UUID, db_context: DatabaseContext, storage: StorageBackend, config: Settings = settings) -> None:
        self.session_id = session_id
        self.db_context = db_context
        self.storage = storage
        self.config = config
        self.token = uuid4()
        self.source: str | None = None

    async def claim(self) -> CalibrationCaptureSession | None:
        async with self.db_context() as db:
            now = utcnow()
            claimed = await db.scalar(update(CalibrationCaptureSession).where(
                CalibrationCaptureSession.id == self.session_id,
                CalibrationCaptureSession.status == "queued",
                CalibrationCaptureSession.deadline_at > now,
                CalibrationCaptureSession.created_at > now - timedelta(seconds=self.config.CAPTURE_QUEUE_TIMEOUT_SEC),
            ).values(status="running", worker_token=self.token, started_at=now, heartbeat_at=now)
                .returning(CalibrationCaptureSession.id))
            if not claimed:
                return None
            return await db.get(CalibrationCaptureSession, self.session_id)

    async def locked_session(self, db: AsyncSession) -> CalibrationCaptureSession | None:
        session = await db.scalar(select(CalibrationCaptureSession).where(
            CalibrationCaptureSession.id == self.session_id,
        ).with_for_update())
        if session is None or session.worker_token != self.token or session.status not in ACTIVE_STATUSES:
            return None
        return session

    async def tick(self, *, attempt: bool = False) -> bool:
        async with self.db_context() as db:
            session = await self.locked_session(db)
            if session is None or expire_session(session, utcnow(), self.config):
                return False
            camera = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(
                Camera.id == session.camera_id,
            ))).one_or_none()
            if camera is None or not camera.stream_enabled or not camera.rtsp_url:
                finish(session, "failed", "stream_disabled", utcnow())
                session.error_code = "capture_stream_disabled"
                return False
            try:
                validate_rtsp_source(camera.rtsp_url, self.config)
            except MediaMTXError as exc:
                finish(session, "failed", "source_invalid", utcnow())
                session.error_code = exc.code
                return False
            if self.source is not None and camera.rtsp_url != self.source:
                finish(session, "failed", "source_changed", utcnow())
                session.error_code = "capture_source_changed"
                return False
            self.source = camera.rtsp_url
            session.heartbeat_at = utcnow()
            if attempt:
                session.attempted += 1
            return True

    async def wait(self, seconds: float) -> bool:
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while loop.time() < end:
            if not await self.tick():
                return False
            await asyncio.sleep(min(1, end - loop.time()))
        return await self.tick()

    async def monitored_frame(self, camera_id: UUID) -> CapturedFrame | None:
        task = asyncio.create_task(capture_frame(camera_id, self.config))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=1)
                if not await self.tick():
                    return None
                if done:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, CaptureError):
                await task

    async def record_failure(self, code: str, terminal: bool) -> None:
        async with self.db_context() as db:
            session = await self.locked_session(db)
            if session is None or expire_session(session, utcnow(), self.config):
                return
            session.failed_attempts += 1
            session.error_code = code
            if terminal:
                finish(session, "failed", "capture_failed", utcnow())

    async def save_frame(self, camera_id: UUID, frame: CapturedFrame) -> bool:
        key = generate_storage_key(str(camera_id), str(self.session_id), "calibration-captures", "jpg")
        discard = True
        try:
            await asyncio.to_thread(self.storage.save, key, io.BytesIO(frame.content), "image/jpeg")
            async with self.db_context() as db:
                session = await self.locked_session(db)
                if session is None or expire_session(session, utcnow(), self.config):
                    return False
                if session.resolution_w is not None and (session.resolution_w, session.resolution_h) != (frame.width, frame.height):
                    session.failed_attempts += 1
                    session.error_code = "capture_resolution_changed"
                    finish(session, "failed", "resolution_changed", utcnow())
                    return False
                session.resolution_w, session.resolution_h = frame.width, frame.height
                session.captured += 1
                session.last_capture_at = frame.captured_at
                session.error_code = None
                db.add(CalibrationCaptureImage(
                    id=uuid4(), session_id=self.session_id, sequence=session.captured,
                    storage_key=key, width=frame.width, height=frame.height,
                    size_bytes=len(frame.content), captured_at=frame.captured_at,
                ))
                if session.captured >= session.target_images:
                    finish(session, "completed", "target_reached", utcnow())
            discard = False
            return True
        finally:
            if discard:
                try:
                    await asyncio.to_thread(self.storage.delete, key)
                except Exception:
                    logger.warning("Capture artifact cleanup failed session=%s", self.session_id)

    async def fail(self, code: str) -> None:
        async with self.db_context() as db:
            session = await self.locked_session(db)
            if session is not None and not expire_session(session, utcnow(), self.config):
                session.error_code = code
                finish(session, "failed", "worker_error", utcnow())

    async def run(self) -> None:
        session = await self.claim()
        if session is None:
            return
        failures = 0
        try:
            while await self.tick(attempt=True):
                try:
                    frame = await self.monitored_frame(session.camera_id)
                    if frame is None or not await self.save_frame(session.camera_id, frame):
                        return
                    failures = 0
                except CaptureError as exc:
                    failures += 1
                    terminal = failures >= self.config.CAPTURE_MAX_CONSECUTIVE_FAILURES or exc.code in {
                        "capture_ffmpeg_missing", "capture_runtime_missing", "stream_service_not_configured", "capture_frame_too_large",
                        "capture_resolution_too_large",
                    }
                    await self.record_failure(exc.code, terminal)
                    if terminal:
                        return
                if not await self.wait(session.interval_sec):
                    return
        except Exception:
            logger.error("Capture worker failed session=%s", self.session_id)
            await self.fail("capture_worker_error")
