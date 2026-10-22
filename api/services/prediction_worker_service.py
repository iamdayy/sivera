import asyncio
import logging
from datetime import datetime, timedelta
from uuid import UUID

import numpy as np
from sqlalchemy import select, update

from api.core.config import Settings, settings
from api.core.database import get_db_context
from api.core.storage import StorageBackend
from api.models.orm import Camera, PredictionJob
from api.services.calibration_capture_service import utcnow
from api.services.image_service import decode_prediction_image
from api.services.prediction_orchestrator import run_prediction_pipeline
from api.services.rtsp_capture_service import CaptureError
from api.services.snapshot_service import capture_snapshot, check_freshness, latest_snapshot, read_snapshot, require_rtsp_camera
from api.workers.database import DatabaseContext


logger = logging.getLogger(__name__)
SAFE_ERRORS = {
    "prediction_invalid_image", "prediction_image_too_large", "prediction_resolution_mismatch",
    "calibration_intrinsics_missing", "prediction_prerequisites_missing", "prediction_source_changed",
    "snapshot_not_found", "snapshot_stale", "snapshot_source_changed", "snapshot_requires_enabled_stream",
    "capture_timeout", "capture_stream_unavailable", "capture_resolution_too_large", "capture_frame_too_large",
    "stream_service_not_configured", "snapshot_image_missing", "snapshot_resolution_mismatch",
    "prediction_invalid_depth", "snapshot_stream_sync_pending",
    "prediction_volume_grid_too_large",
    "calibration_invalid_intrinsics",
}


class PredictionRunner:
    def __init__(self, job_id: UUID, db_context: DatabaseContext, storage: StorageBackend, config: Settings = settings) -> None:
        self.job_id, self.db_context, self.storage, self.config = job_id, db_context, storage, config

    async def load_image(self, job: PredictionJob) -> tuple[np.ndarray, datetime | None, str]:
        if job.source_type == "upload":
            def load():
                stream = self.storage.get(job.source_path)
                if stream is None:
                    raise ValueError("prediction_invalid_image")
                try:
                    return decode_prediction_image(stream.read(self.config.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1), self.config)
                finally:
                    stream.close()
            return await asyncio.to_thread(load), job.source_ts, job.source_path
        async with self.db_context() as db:
            camera = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(Camera.id == job.camera_id))).one_or_none()
            if camera is None:
                raise ValueError("prediction_prerequisites_missing")
            identity = require_rtsp_camera(camera, self.config)
            if job.source_hash != identity:
                raise ValueError("prediction_source_changed")
            snapshot = await latest_snapshot(db, job.camera_id, identity, job.max_frame_age_sec) if job.source_type == "latest_rtsp" else None
        if snapshot is None:
            snapshot = await capture_snapshot(job.camera_id, self.db_context, self.storage, self.config, identity)
        frame = await asyncio.to_thread(read_snapshot, self.storage, snapshot, self.config)
        check_freshness(frame.captured_at, job.max_frame_age_sec)
        return decode_prediction_image(frame.content, self.config), frame.captured_at, snapshot.storage_key

    async def run(self) -> dict | None:
        async with self.db_context() as db:
            claimed = await db.scalar(update(PredictionJob).where(PredictionJob.id == self.job_id, PredictionJob.status == "pending").values(
                status="processing", started_at=utcnow(), error_msg=None,
            ).returning(PredictionJob.id))
            if claimed is None:
                return None
            job = await db.get(PredictionJob, self.job_id)
        try:
            image, timestamp, path = await self.load_image(job)
            async with self.db_context() as db:
                await db.execute(update(PredictionJob).where(PredictionJob.id == self.job_id, PredictionJob.status == "processing").values(source_ts=timestamp, source_path=path))
            async with self.db_context() as db:
                job = await db.get(PredictionJob, self.job_id)
                if job is None or job.status != "processing":
                    return None
                summary = await run_prediction_pipeline(db, job, image)
                # Avoid overwriting a timeout/failure that fenced this worker during inference.
                with db.no_autoflush:
                    if job.source_type != "upload":
                        camera = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(
                            Camera.id == job.camera_id,
                        ).with_for_update())).one_or_none()
                        if camera is None or require_rtsp_camera(camera, self.config) != job.source_hash:
                            raise ValueError("prediction_source_changed")
                    status = await db.scalar(select(PredictionJob.status).where(PredictionJob.id == self.job_id).with_for_update())
                if status != "processing":
                    await db.rollback()
                    return None
                job.status, job.result, job.completed_at = "done", summary, utcnow()
            return summary
        except Exception as exc:
            value = exc.code if isinstance(exc, CaptureError) else str(exc) if isinstance(exc, ValueError) else ""
            code = value if value in SAFE_ERRORS else "prediction_processing_failed"
            logger.warning("Prediction failed job=%s code=%s", self.job_id, code)
            async with self.db_context() as db:
                await db.execute(update(PredictionJob).where(PredictionJob.id == self.job_id, PredictionJob.status == "processing").values(
                    status="failed", error_msg=code, completed_at=utcnow(), result=None,
                ))
            return None


async def reconcile_predictions(config: Settings = settings) -> None:
    try:
        async with get_db_context() as db:
            now = utcnow()
            await db.execute(update(PredictionJob).where(
                PredictionJob.status == "pending", PredictionJob.created_at <= now - timedelta(seconds=config.CAPTURE_QUEUE_TIMEOUT_SEC),
            ).values(status="failed", error_msg="prediction_queue_timeout", completed_at=now))
            await db.execute(update(PredictionJob).where(
                PredictionJob.status == "processing", PredictionJob.started_at <= now - timedelta(seconds=config.CELERY_TASK_TIME_LIMIT + 60),
            ).values(status="failed", error_msg="prediction_worker_timeout", completed_at=now))
    except Exception:
        logger.error("Prediction reconciliation failed; check database configuration")
