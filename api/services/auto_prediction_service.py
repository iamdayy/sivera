import asyncio
import base64
import logging
from datetime import timedelta
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import raiseload

from api.core.config import Settings, settings
from api.core.database import DatabaseContext
from api.core.storage import StorageBackend
from api.models.orm import Camera, PredictionJob
from api.schemas.auto_prediction import AutoPredictConfig
from api.services.calibration_capture_service import utcnow
from api.services.live_capture_service import capture_live_frame
from api.services.prediction_orchestrator import active_prerequisites, measurement_context
from api.services.prediction_result_service import prediction_error
from api.services.prediction_result_service import PredictionArtifacts, finish_io, grid_volume, last_saved_volume
from api.services.snapshot_service import check_freshness, source_hash
from api.services.vision_process_service import run_vision, vision_slot


logger = logging.getLogger(__name__)


def reset_auto_schedule(camera: Camera) -> None:
    camera.auto_predict_run_token = None
    camera.auto_predict_deadline_at = None
    camera.auto_predict_next_run = utcnow() if camera.stream_enabled and (camera.auto_predict_config or {}).get("enabled") else None


class AutoPredictionRunner:
    def __init__(self, camera_id: UUID, db_context: DatabaseContext, storage: StorageBackend, config: Settings = settings) -> None:
        self.camera_id, self.db_context, self.storage, self.config = camera_id, db_context, storage, config
        self.token = str(uuid4())
        self.identity: str | None = None
        self.options: dict = {}

    async def claim(self) -> Camera | None:
        async with self.db_context() as db:
            camera = await db.scalar(select(Camera).options(raiseload("*")).where(
                Camera.id == self.camera_id,
            ).with_for_update())
            now = utcnow()
            if (camera is None or not camera.stream_enabled or not camera.rtsp_url
                    or not (camera.auto_predict_config or {}).get("enabled") or camera.auto_predict_run_token
                    or camera.auto_predict_next_run is not None and camera.auto_predict_next_run > now):
                return None
            self.options = AutoPredictConfig.model_validate(camera.auto_predict_config).model_dump()
            self.identity = source_hash(camera.rtsp_url, self.config)
            camera.auto_predict_run_token = self.token
            camera.auto_predict_deadline_at = now + timedelta(seconds=self.config.PREDICTION_TIMEOUT_SEC)
            # Overdue intervals are one due timestamp, never a backlog of captured images.
            camera.auto_predict_next_run = now + timedelta(seconds=self.options["interval_sec"])
            return camera

    def current(self, camera: Camera, *, check_deadline: bool = True) -> bool:
        return bool(camera is not None and camera.auto_predict_run_token == self.token
                    and camera.stream_enabled and camera.rtsp_url
                    and (camera.auto_predict_config or {}).get("enabled")
                    and source_hash(camera.rtsp_url, self.config) == self.identity
                    and (not check_deadline or camera.auto_predict_deadline_at is not None
                         and camera.auto_predict_deadline_at > utcnow()))

    async def active(self) -> bool:
        async with self.db_context() as db:
            camera = await db.scalar(select(Camera).options(raiseload("*")).where(Camera.id == self.camera_id))
            return self.current(camera)

    async def finish(self, event: dict) -> bool:
        async with self.db_context() as db:
            camera = await db.scalar(select(Camera).options(raiseload("*")).where(
                Camera.id == self.camera_id,
            ).with_for_update())
            if not self.current(camera, check_deadline=False):
                return False
            camera.auto_predict_run_token = None
            camera.auto_predict_deadline_at = None
            camera.auto_predict_last_run_at = utcnow()
            camera.auto_predict_last_run_status = "saved" if event["saved"] else "failed" if event["error_code"] else "skipped"
            camera.auto_predict_last_run_error = event["error_code"]
        return True

    async def _publish(self, event: dict) -> None:
        """Publish event ke SSE subscribers (fire-and-forget)."""
        from api.runtime import event_hub
        try:
            await event_hub.publish(self.camera_id, event)
        except Exception:
            # Event hub tidak kritis untuk alur prediksi; abaikan gagal publish.
            pass

    async def _current_in_db(self, db) -> bool:
        camera = await db.scalar(select(Camera).options(raiseload("*")).where(
            Camera.id == self.camera_id,
        ))
        return self.current(camera)

    async def persist(self, context: dict, result: dict, event: dict) -> None:
        volume = grid_volume(result)
        event["volume_liters"], event["volume_method"] = volume, "grid" if volume is not None else None
        if volume is None:
            event["skip_reason"] = "no_valid_volume"
            return
        async with self.db_context() as db:
            previous = await last_saved_volume(db, self.camera_id)
        if previous == volume:
            event["skip_reason"] = "volume_unchanged"
            return
        if not await self.active():
            return
        job_id = uuid4()
        artifacts = PredictionArtifacts(self.storage, self.camera_id, job_id)

        async def commit() -> bool:
            async with self.db_context() as db:
                camera = await db.scalar(select(Camera).options(raiseload("*")).where(
                    Camera.id == self.camera_id,
                ).with_for_update())
                if not self.current(camera):
                    return False
                calibration, ground, scale = await active_prerequisites(db, self.camera_id, "rtsp_frame")
                if context["ids"] != {"calibration_id": str(calibration.id), "ground_model_id": str(ground.id), "depth_scale_id": str(scale.id)}:
                    raise ValueError("prediction_prerequisites_changed")
                if await last_saved_volume(db, self.camera_id) == volume:
                    event["skip_reason"] = "volume_unchanged"
                    return False
                job = PredictionJob(
                    id=job_id, camera_id=self.camera_id, **{key: UUID(value) for key, value in context["ids"].items()},
                    status="done", source_type="rtsp_frame", source_path=None,
                    source_ts=event["captured_at"], source_hash=self.identity,
                    started_at=event["started_at"], completed_at=utcnow(),
                    result=result["summary"], **{key: self.options[key] for key in (
                        "seg_model", "seg_conf", "seg_classes", "volume_cell_m", "volume_min_pts_per_cell",
                        "return_outputs", "max_frame_age_sec",
                    )},
                )
                db.add(job)
                await db.flush()
                artifacts.add_rows(db, result)
                camera.auto_predict_last_saved_job_id = job_id
            artifacts.committed = True
            event["saved"], event["job_id"] = True, str(job_id)
            return True

        try:
            await artifacts.write(result["artifacts"], self.options["return_outputs"])
            await finish_io(commit())
        finally:
            await artifacts.cleanup()

    async def run(self) -> dict | None:
        camera = await self.claim()
        if camera is None:
            return None
        event = {
            "run_id": self.token, "camera_id": str(self.camera_id), "started_at": utcnow(),
            "captured_at": None, "completed_at": None, "frame_w": 0, "frame_h": 0, "frame_jpeg_base64": None,
            "segmentation": {"has_mask": False, "waste_fraction_pct": None, "mask_png_base64": None,
                             "overlay_png_base64": None, "coordinate_space": "raw"},
            "volume_liters": None, "volume_method": None, "saved": False, "job_id": None,
            "skip_reason": None, "error_code": None,
        }
        result, context = None, None
        segmentation = None
        try:
            async with asyncio.timeout(self.config.PREDICTION_TIMEOUT_SEC):
                # Stage 1: capture frame + segment original image (vision slot only).
                async with vision_slot(self.config.PREDICTION_TIMEOUT_SEC, self.active):
                    if not await self.active():
                        return None
                    frame = await capture_live_frame(self.camera_id, self.config, self.identity, self.db_context)
                    check_freshness(frame.captured_at, self.options["max_frame_age_sec"])
                    image_b64 = base64.b64encode(frame.content).decode("ascii")
                    event.update(captured_at=frame.captured_at, frame_w=frame.width, frame_h=frame.height,
                                 frame_jpeg_base64=image_b64)
                    segmentation = await run_vision("segment", {"image": image_b64, "options": self.options},
                                                    self.config.PREDICTION_TIMEOUT_SEC, self.active)
                    event["segmentation"] = {**segmentation, "coordinate_space": "raw"}

                # Jika tidak ada waste mask: segera kirim frame asli (tidak ada measurement).
                if not segmentation or not segmentation.get("has_mask"):
                    event["skip_reason"] = "no_segmentation"
                    await self._publish(event)
                else:
                    # Segmenasi terdeteksi: segera kirim frame + mask overlay ke frontend.
                    await self._publish(event)

                    # Stage 2: resolve prerequisite (DB only, tanpa vision slot).
                    async with self.db_context() as db:
                        if not await self._current_in_db(db):
                            return None
                        calibration, ground, scale = await active_prerequisites(db, self.camera_id, "rtsp_frame")
                        context = measurement_context(camera, calibration, ground, scale)

                    # Stage 3: measure + volume (vision slot).
                    async with vision_slot(self.config.PREDICTION_TIMEOUT_SEC, self.active):
                        if not await self.active():
                            return None
                        result = await run_vision("measure", {
                            "image": event["frame_jpeg_base64"],
                            "segmentation": segmentation, "context": context, "options": self.options,
                        }, self.config.PREDICTION_TIMEOUT_SEC, self.active)

                    if result is not None:
                        await self.persist(context, result, event)
                        # Publish update akhir (volume/saved) setelah measurement.
                        await self._publish(event)
        except asyncio.CancelledError:
            event["error_code"] = "prediction_interrupted"
            await self._publish(event)
            await self.finish(event)
            raise
        except Exception as exc:
            code = "prediction_prerequisites_changed" if str(exc) == "prediction_prerequisites_changed" else prediction_error(exc)
            event["skip_reason"], event["error_code"] = code, code
            logger.warning("Automatic prediction failed camera=%s run=%s code=%s exception=%s",
                           self.camera_id, self.token, code, type(exc).__name__)
        event["completed_at"] = utcnow()
        if not await self.finish(event):
            return None
        for key in ("captured_at", "started_at", "completed_at"):
            event[key] = event[key].isoformat() if event[key] else None
        return event
