from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.core.config import settings


DatabaseContext = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@asynccontextmanager
async def task_database() -> AsyncIterator[DatabaseContext]:
    # Each prefork task owns its engine and event loop, including redeliveries.
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool, hide_parameters=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def context() -> AsyncIterator[AsyncSession]:
        async with factory() as db:
            try:
                yield db
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    try:
        yield context
    finally:
        await engine.dispose()
