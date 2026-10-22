import asyncio
from uuid import UUID

from api.core.storage import get_storage
from api.services.calibration_compute_service import CalibrationComputeRunner
from api.workers.celery_app import celery_app
from api.workers.database import task_database


@celery_app.task(ignore_result=True, acks_late=True, reject_on_worker_lost=True, time_limit=660, max_retries=0)
def run_capture_calibration_task(session_id: str) -> None:
    asyncio.run(run_calibration(UUID(session_id)))


async def run_calibration(session_id: UUID) -> None:
    async with task_database() as context:
        await CalibrationComputeRunner(session_id, context, get_storage()).run()
