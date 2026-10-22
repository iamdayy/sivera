import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.core.config import settings
from api.core.storage import get_storage
from api.services.calibration_capture_service import CaptureRunner
from api.workers.celery_app import celery_app


@celery_app.task(
    name="api.workers.tasks.capture_task.run_capture_task", ignore_result=True,
    acks_late=True, reject_on_worker_lost=True, time_limit=960, max_retries=0,
)
def run_capture_task(session_id: str) -> None:
    asyncio.run(run_capture(UUID(session_id)))


async def run_capture(session_id: UUID) -> None:
    # Prefork children and successive asyncio.run calls cannot share pooled async connections.
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool, hide_parameters=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def context() -> AsyncIterator[AsyncSession]:
        async with factory() as db:
            try:
                yield db
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    try:
        runner = CaptureRunner(session_id, context, get_storage())
        await runner.run()
    finally:
        await engine.dispose()
