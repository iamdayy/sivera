import asyncio
from uuid import UUID

from api.core.storage import get_storage
from api.services.prediction_worker_service import PredictionRunner
from api.workers.celery_app import celery_app
from api.workers.database import task_database


@celery_app.task(acks_late=True, reject_on_worker_lost=True, max_retries=0)
def run_prediction_task(job_id: str) -> dict | None:
    return asyncio.run(run_prediction(UUID(job_id)))


async def run_prediction(job_id: UUID) -> dict | None:
    async with task_database() as context:
        return await PredictionRunner(job_id, context, get_storage()).run()


@celery_app.task(bind=True, max_retries=0)
def run_calibration_task(self, camera_id: str, image_files: list, cols: int, rows: int, square_m: float, alpha: float, notes: str):
    async def run():
        from api.models.orm import Camera
        from api.services.calibration_service import process_calibration_upload
        async with task_database() as context, context() as db:
            camera = await db.get(Camera, UUID(camera_id))
            if camera is None:
                return None
            calibration = await process_calibration_upload(db, camera, image_files, cols, rows, square_m, alpha, notes)
            return str(calibration.id)
    return asyncio.run(run())
