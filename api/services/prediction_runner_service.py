import asyncio
import base64
import logging
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.orm import raiseload

from api.core.config import Settings, settings
from api.core.database import DatabaseContext
from api.core.storage import StorageBackend
from api.models.orm import Camera, PredictionJob
from api.services.calibration_capture_service import utcnow
from api.services.live_capture_service import capture_live_frame
from api.services.prediction_codes import SAFE_ERRORS, prediction_error
from api.services.prediction_orchestrator import pinned_context, prediction_options
from api.services.prediction_result_service import PredictionArtifacts, finish_io
from api.services.snapshot_service import check_freshness, latest_snapshot, read_snapshot, require_rtsp_camera
from api.services.vision_process_service import VisionError, run_vision, vision_slot


logger = logging.getLogger(__name__)


class PredictionRunner:
    def __init__(self, job_id: UUID, db_context: DatabaseContext, storage: StorageBackend, config: Settings = settings) -> None:
        self.job_id, self.db_context, self.storage, self.config = job_id, db_context, storage, config
        self.token = uuid4()

    async def claim(self) -> PredictionJob | None:
        async with self.db_context() as db:
            now = utcnow()
            claimed = await db.scalar(update(PredictionJob).where(
                PredictionJob.id == self.job_id, PredictionJob.status == "pending",
                PredictionJob.created_at > now - timedelta(seconds=self.config.CAPTURE_QUEUE_TIMEOUT_SEC),
                or_(PredictionJob.deadline_at.is_(None), PredictionJob.deadline_at > now),
            ).values(status="processing", started_at=now, run_token=self.token, error_msg=None,
                     deadline_at=now + timedelta(seconds=self.config.PREDICTION_TIMEOUT_SEC)).returning(PredictionJob.id))
            return await db.get(PredictionJob, self.job_id, options=[raiseload("*")]) if claimed else None

    async def active(self) -> bool:
        async with self.db_context() as db:
            return await db.scalar(select(PredictionJob.id).where(
                PredictionJob.id == self.job_id, PredictionJob.status == "processing",
                PredictionJob.run_token == self.token, PredictionJob.deadline_at > utcnow(),
            )) is not None

    async def load_image(self, job: PredictionJob) -> tuple[bytes, datetime | None, str | None]:
        if job.source_type == "upload":
            def load():
                stream = self.storage.get(job.source_path)
                if stream is None:
                    raise ValueError("prediction_invalid_image")
                try:
                    content = stream.read(self.config.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
                finally:
                    stream.close()
                if len(content) > self.config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    raise ValueError("prediction_image_too_large")
                return content
            return await asyncio.to_thread(load), job.source_ts, job.source_path
        async with self.db_context() as db:
            camera = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(Camera.id == job.camera_id))).one_or_none()
            if camera is None:
                raise ValueError("prediction_prerequisites_missing")
            identity = require_rtsp_camera(camera, self.config)
            if job.source_hash != identity:
                raise ValueError("prediction_source_changed")
            snapshot = await latest_snapshot(db, job.camera_id, identity, job.max_frame_age_sec) if job.source_type == "latest_rtsp" else None
        if snapshot is not None:
            frame = await asyncio.to_thread(read_snapshot, self.storage, snapshot, self.config)
        else:
            frame = await capture_live_frame(job.camera_id, self.config, identity, self.db_context)
        check_freshness(frame.captured_at, job.max_frame_age_sec)
        return frame.content, frame.captured_at, snapshot.storage_key if snapshot else None

    async def complete(self, job: PredictionJob, result: dict, artifacts: PredictionArtifacts,
                       timestamp: datetime | None, path: str | None) -> bool:
        async with self.db_context() as db:
            camera = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(
                Camera.id == job.camera_id,
            ).with_for_update())).one_or_none()
            current = await db.scalar(select(PredictionJob).options(raiseload("*")).where(
                PredictionJob.id == self.job_id,
            ).with_for_update())
            if current is None or current.status != "processing" or current.run_token != self.token:
                return False
            if current.deadline_at <= utcnow():
                raise ValueError("prediction_timeout")
            if camera is None or job.source_type != "upload" and require_rtsp_camera(camera, self.config) != job.source_hash:
                raise ValueError("prediction_source_changed")
            artifacts.add_rows(db, result)
            current.status, current.result, current.completed_at = "done", result["summary"], utcnow()
            current.source_ts, current.source_path, current.run_token = timestamp, path, None
        artifacts.committed = True
        return True

    async def fail(self, code: str) -> None:
        async with self.db_context() as db:
            await db.execute(update(PredictionJob).where(
                PredictionJob.id == self.job_id, PredictionJob.status == "processing", PredictionJob.run_token == self.token,
            ).values(status="failed", error_msg=code, completed_at=utcnow(), result=None, run_token=None))

    async def run(self) -> dict | None:
        job = await self.claim()
        if job is None:
            return None
        artifacts = PredictionArtifacts(self.storage, job.camera_id, job.id)
        try:
            async with asyncio.timeout(self.config.PREDICTION_TIMEOUT_SEC):
                async with vision_slot(self.config.PREDICTION_TIMEOUT_SEC, self.active):
                    async with self.db_context() as db:
                        context = await pinned_context(db, job)
                    image, timestamp, path = await self.load_image(job)
                    result = await run_vision("predict", {
                        "image": base64.b64encode(image).decode("ascii"), "options": prediction_options(job), "context": context,
                    }, self.config.PREDICTION_TIMEOUT_SEC, self.active)
                if not await self.active():
                    return None
                await artifacts.write(result["artifacts"], job.return_outputs)
                if await finish_io(self.complete(job, result, artifacts, timestamp, path)):
                    return result["summary"]
        except asyncio.CancelledError:
            await self.fail("prediction_interrupted")
            raise
        except Exception as exc:
            code = prediction_error(exc)
            logger.warning("Prediction failed job=%s code=%s exception=%s", self.job_id, code, type(exc).__name__)
            await self.fail(code)
        finally:
            await artifacts.cleanup()
        return None
