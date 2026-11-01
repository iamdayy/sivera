from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import Settings, settings
from api.models.orm import CalibrationCaptureSession, PredictionJob


async def admit_job(db: AsyncSession, config: Settings = settings) -> None:
    # Serialize admission until the caller commits its durable domain row.
    await db.execute(text("SELECT pg_advisory_xact_lock(1397315137)"))
    predictions = select(func.count()).select_from(PredictionJob).where(PredictionJob.status.in_(("pending", "processing"))).scalar_subquery()
    captures = select(func.count()).select_from(CalibrationCaptureSession).where(CalibrationCaptureSession.status.in_(("queued", "running", "stopping"))).scalar_subquery()
    computations = select(func.count()).select_from(CalibrationCaptureSession).where(CalibrationCaptureSession.compute_status.in_(("queued", "running"))).scalar_subquery()
    if await db.scalar(select(predictions + captures + computations)) >= config.RUNTIME_MAX_PENDING_JOBS:
        raise HTTPException(status_code=503, detail={"code": "runtime_queue_full"}, headers={"Retry-After": "5"})
