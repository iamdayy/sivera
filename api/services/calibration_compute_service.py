import asyncio
import logging
from datetime import timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import Settings, settings
from api.core.database import get_db_context
from api.core.storage import StorageBackend
from api.models.orm import Calibration, CalibrationCaptureImage, CalibrationCaptureSession, Camera
from api.services.calibration_capture_service import ACTIVE_STATUSES, DISPATCH_INTERVAL_SEC, utcnow
from api.services.calibration_quality_service import assess_frame_bounded
from api.services.rtsp_capture_service import decode_frame
from api.services.vision_process_service import VisionError, run_vision
from api.workers.database import DatabaseContext


logger = logging.getLogger(__name__)


def fail_compute(session: CalibrationCaptureSession, code: str) -> None:
    session.compute_status = "failed"
    session.compute_error_code = code
    session.compute_completed_at = utcnow()
    session.compute_token = None


def expire_computation(session: CalibrationCaptureSession, config: Settings = settings) -> None:
    now = utcnow()
    if session.compute_status == "queued" and session.compute_requested_at is not None and now >= session.compute_requested_at + timedelta(seconds=config.CAPTURE_QUEUE_TIMEOUT_SEC):
        fail_compute(session, "calibration_queue_timeout")
    elif session.compute_status == "running" and session.compute_started_at is not None and now >= session.compute_started_at + timedelta(seconds=config.CALIB_COMPUTE_TIMEOUT_SEC):
        fail_compute(session, "calibration_compute_timeout")


async def publish_computation(session_id: UUID) -> None:
    from api.workers.tasks.calibration_task import run_capture_calibration_task
    await asyncio.to_thread(run_capture_calibration_task.apply_async, args=[str(session_id)], queue="calibration", retry=False)


async def reconcile_computations(config: Settings = settings) -> None:
    try:
        dispatch = []
        async with get_db_context() as db:
            sessions = (await db.execute(select(CalibrationCaptureSession).where(
                CalibrationCaptureSession.compute_status.in_(("queued", "running")),
            ).with_for_update(skip_locked=True))).scalars().all()
            now = utcnow()
            for session in sessions:
                expire_computation(session, config)
                if session.compute_status == "queued" and (session.compute_dispatched_at is None or session.compute_dispatched_at <= now - timedelta(seconds=DISPATCH_INTERVAL_SEC)):
                    session.compute_dispatched_at = now
                    dispatch.append(session.id)
        for session_id in dispatch:
            try:
                await publish_computation(session_id)
            except Exception:
                logger.warning("Calibration dispatch deferred session=%s", session_id)
    except Exception:
        logger.error("Calibration reconciliation failed; check database and queue configuration")


class CalibrationComputeRunner:
    def __init__(self, session_id: UUID, db_context: DatabaseContext, storage: StorageBackend, config: Settings = settings) -> None:
        self.session_id, self.db_context, self.storage, self.config = session_id, db_context, storage, config
        self.token = uuid4()

    async def claim(self) -> CalibrationCaptureSession | None:
        async with self.db_context() as db:
            claimed = await db.scalar(update(CalibrationCaptureSession).where(
                CalibrationCaptureSession.id == self.session_id,
                CalibrationCaptureSession.compute_status == "queued",
                CalibrationCaptureSession.status.not_in(ACTIVE_STATUSES),
                CalibrationCaptureSession.compute_requested_at > utcnow() - timedelta(seconds=self.config.CAPTURE_QUEUE_TIMEOUT_SEC),
            ).values(compute_status="running", compute_token=self.token, compute_started_at=utcnow()).returning(CalibrationCaptureSession.id))
            return await db.get(CalibrationCaptureSession, self.session_id) if claimed else None

    async def locked(self, db: AsyncSession) -> CalibrationCaptureSession | None:
        session = await db.scalar(select(CalibrationCaptureSession).where(CalibrationCaptureSession.id == self.session_id).with_for_update())
        if session is None or session.compute_token != self.token or session.compute_status != "running":
            return None
        expire_computation(session, self.config)
        return session if session.compute_status == "running" else None

    async def active(self) -> bool:
        async with self.db_context() as db:
            session = await self.locked(db)
            if session is None:
                return False
            if session.source_hash is not None:
                from api.services.snapshot_service import source_hash
                source = await db.scalar(select(Camera.rtsp_url).where(Camera.id == session.camera_id))
                if source is None or source_hash(source, self.config) != session.source_hash:
                    fail_compute(session, "calibration_source_changed")
                    return False
            return True

    async def compute(self, session: CalibrationCaptureSession) -> None:
        async with self.db_context() as db:
            images = (await db.execute(select(CalibrationCaptureImage).where(CalibrationCaptureImage.session_id == self.session_id).order_by(CalibrationCaptureImage.sequence).limit(301))).scalars().all()
        if len(images) > 300:
            raise VisionError("calibration_frame_limit")
        previous, selected = [], []
        for image in images:
            if image.quality_status == "unassessed":
                def load():
                    stream = self.storage.get(image.storage_key)
                    if stream is None:
                        raise VisionError("calibration_image_missing")
                    try:
                        return decode_frame(stream.read(self.config.CAPTURE_MAX_FRAME_BYTES + 1), self.config)
                    finally:
                        stream.close()
                frame = await asyncio.to_thread(load)
                if (frame.width, frame.height) != (image.width, image.height):
                    raise VisionError("calibration_resolution_mismatch")
                quality = await assess_frame_bounded(frame, session.chessboard_cols, session.chessboard_rows, previous, self.config, self.active)
                async with self.db_context() as db:
                    current = await self.locked(db)
                    if current is None:
                        return
                    row = await db.get(CalibrationCaptureImage, image.id)
                    row.quality_status, row.rejection_reason, row.sharpness = quality.status, quality.reason, quality.sharpness
                    row.corners, row.pose_descriptor = quality.corners, quality.descriptor
                    if quality.status == "accepted":
                        current.accepted += 1
                    else:
                        current.rejected += 1
                        counts = dict(current.rejection_counts)
                        counts[quality.reason] = counts.get(quality.reason, 0) + 1
                        current.rejection_counts = counts
                image.quality_status, image.corners, image.pose_descriptor = quality.status, quality.corners, quality.descriptor
            if image.quality_status == "accepted":
                previous.append(image.pose_descriptor)
                selected.append(image)
        if len(selected) < self.config.MIN_CALIB_IMAGES:
            raise VisionError("calibration_insufficient_images")
        size = (session.resolution_w, session.resolution_h)
        if not all(size) or any((image.width, image.height) != size for image in selected):
            raise VisionError("calibration_resolution_mismatch")
        result = await run_vision("calibrate", {
            "points": [image.corners for image in selected], "size": size,
            "cols": session.chessboard_cols, "rows": session.chessboard_rows, "square_m": session.square_m,
        }, self.config.CALIB_COMPUTE_TIMEOUT_SEC, self.active)
        async with self.db_context() as db:
            # Lock camera first everywhere version allocation/activation is serialized.
            camera_id = await db.scalar(select(Camera.id).where(Camera.id == session.camera_id).with_for_update())
            if camera_id is None:
                return
            current = await self.locked(db)
            if current is None:
                return
            if current.source_hash is not None:
                from api.services.snapshot_service import source_hash
                source = await db.scalar(select(Camera.rtsp_url).where(Camera.id == camera_id))
                if source is None or source_hash(source, self.config) != current.source_hash:
                    fail_compute(current, "calibration_source_changed")
                    return
            version = (await db.scalar(select(func.max(Calibration.version)).where(Calibration.camera_id == camera_id)) or 0) + 1
            calibration = Calibration(
                id=uuid4(), camera_id=camera_id, version=version, is_active=False,
                k_matrix=result["K_undist"], k_original=result["K"], k_undist=result["K_undist"],
                dist_coeffs=result["dist"], resolution_w=size[0], resolution_h=size[1], alpha=result["alpha"],
                per_view_errors=result["per_view"], rms_reproj=result["rms"],
                coverage_w=result["coverage"][0] * 100, coverage_h=result["coverage"][1] * 100,
                chessboard_cols=session.chessboard_cols, chessboard_rows=session.chessboard_rows,
                square_m=session.square_m, calib_images=[image.storage_key for image in selected],
            )
            db.add(calibration)
            await db.flush()
            current.result_calibration_id = calibration.id
            current.compute_status, current.compute_completed_at, current.compute_token = "done", utcnow(), None

    async def run(self) -> None:
        session = await self.claim()
        if session is None:
            return
        try:
            async with asyncio.timeout(self.config.CALIB_COMPUTE_TIMEOUT_SEC):
                await self.compute(session)
        except Exception as exc:
            codes = {"calibration_insufficient_images", "calibration_invalid_corners", "calibration_invalid_result", "calibration_resolution_mismatch", "calibration_image_missing", "calibration_frame_limit", "calibration_invalid_parameters"}
            code = exc.code if isinstance(exc, VisionError) and exc.code in codes else "calibration_compute_failed"
            if isinstance(exc, TimeoutError) or isinstance(exc, VisionError) and exc.code == "vision_timeout":
                code = "calibration_compute_timeout"
            logger.warning("Calibration computation failed session=%s code=%s", self.session_id, code)
            async with self.db_context() as db:
                current = await self.locked(db)
                if current is not None:
                    fail_compute(current, code)
