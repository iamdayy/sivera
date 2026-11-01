import asyncio
import json
import logging
from contextlib import suppress
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import HTTPException, Request
from sqlalchemy import func, or_, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import raiseload

from api.core.config import Settings, settings
from api.core.database import DatabaseContext
from api.core.storage import StorageBackend
from api.models.orm import CalibrationCaptureSession, Camera, PredictionJob
from api.services.auto_prediction_service import AutoPredictionRunner, reset_auto_schedule
from api.services.calibration_capture_service import ACTIVE_STATUSES, CaptureRunner, expire_session, finish, utcnow
from api.services.calibration_compute_service import CalibrationComputeRunner, expire_computation, fail_compute
from api.services.prediction_runner_service import PredictionRunner


class EventHub:
    def __init__(self, max_clients: int) -> None:
        self.max_clients = max_clients
        self.subscribers: dict[UUID, set[asyncio.Queue]] = {}
        self.closed = False

    @property
    def client_count(self) -> int:
        return sum(len(queues) for queues in self.subscribers.values())

    def subscribe(self, camera_id: UUID) -> asyncio.Queue:
        if self.closed:
            raise ValueError("prediction_stream_unavailable")
        if self.client_count >= self.max_clients:
            raise ValueError("prediction_stream_capacity")
        queue = asyncio.Queue(maxsize=1)
        self.subscribers.setdefault(camera_id, set()).add(queue)
        return queue

    def unsubscribe(self, camera_id: UUID, queue: asyncio.Queue) -> None:
        queues = self.subscribers.get(camera_id)
        if queues is not None:
            queues.discard(queue)
            if not queues:
                del self.subscribers[camera_id]

    @staticmethod
    def offer(queue: asyncio.Queue, value: str | None) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(value)

    async def publish(self, camera_id: UUID, event: dict) -> None:
        if self.closed or camera_id not in self.subscribers:
            return
        payload = await asyncio.to_thread(json.dumps, event, allow_nan=False, separators=(",", ":"))
        for queue in tuple(self.subscribers.get(camera_id, ())):
            self.offer(queue, payload)

    def close(self) -> None:
        self.closed = True
        for queues in self.subscribers.values():
            for queue in queues:
                self.offer(queue, None)
        self.subscribers.clear()


if TYPE_CHECKING:
    from api.runtime.manager import LocalRuntime


def require_runtime(request: Request) -> "LocalRuntime":
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.status != "running":
        raise HTTPException(status_code=503, detail={"code": "runtime_unavailable"}, headers={"Retry-After": "5"})
    return runtime


def notify_runtime(request: Request | None = None) -> None:
    if request is not None:
        runtime = getattr(request.app.state, "runtime", None)
    else:
        runtime = None
    if runtime is not None:
        runtime.notify()


async def admit_job(db: AsyncSession, config: Settings = settings) -> None:
    # Serialize admission until the caller commits its durable domain row.
    await db.execute(text("SELECT pg_advisory_xact_lock(1397315137)"))
    predictions = select(func.count()).select_from(PredictionJob).where(PredictionJob.status.in_(("pending", "processing"))).scalar_subquery()
    captures = select(func.count()).select_from(CalibrationCaptureSession).where(CalibrationCaptureSession.status.in_(("queued", "running", "stopping"))).scalar_subquery()
    computations = select(func.count()).select_from(CalibrationCaptureSession).where(CalibrationCaptureSession.compute_status.in_(("queued", "running"))).scalar_subquery()
    total = await db.scalar(select(predictions + captures + computations))
    if total >= config.RUNTIME_MAX_PENDING_JOBS:
        logger.error("Runtime queue full pending=%d max=%d", total, config.RUNTIME_MAX_PENDING_JOBS)
        raise HTTPException(status_code=503, detail={"code": "runtime_queue_full"}, headers={"Retry-After": "5"})


logger = logging.getLogger(__name__)
RUNTIME_LOCK = 0x534956455241
WORK_KINDS = ("prediction", "calibration", "automatic")


class LocalRuntime:
    def __init__(self, engine: AsyncEngine, db_context: DatabaseContext, storage: StorageBackend,
                 events: EventHub, config: Settings = settings) -> None:
        self.engine, self.db_context, self.storage, self.events, self.config = engine, db_context, storage, events, config
        self.status = "stopped"
        self.wakeup = asyncio.Event()
        self.connection = None
        self.backend_pid: int | None = None
        self.loop_task: asyncio.Task | None = None
        self.work_task: asyncio.Task | None = None
        self.capture_task: asyncio.Task | None = None
        self.work_kind: str | None = None
        self.cursor = 0

    def notify(self) -> None:
        self.wakeup.set()

    async def start(self) -> None:
        if not self.config.RUNTIME_ENABLED:
            self.status = "disabled"
            return
        self.connection = await self.engine.connect()
        try:
            await self.connection.execution_options(isolation_level="AUTOCOMMIT")
            acquired = await self.connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": RUNTIME_LOCK})
            if not acquired:
                raise RuntimeError("runtime_already_running: use one API process and replica")
            self.backend_pid = await self.connection.scalar(text("SELECT pg_backend_pid()"))
            await self.reconcile(interrupted=True)
        except BaseException:
            await self.release_lock()
            raise
        self.status = "running"
        self.loop_task = asyncio.create_task(self.loop(), name="sivera-runtime")
        logger.info("Local runtime started vision_capacity=1 capture_sessions=1")

    async def release_lock(self) -> None:
        if self.connection is not None:
            try:
                if not self.connection.invalidated:
                    await self.connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": RUNTIME_LOCK})
            finally:
                await self.connection.close()
                self.connection = None

    async def reconcile(self, *, interrupted: bool = False) -> None:
        async with self.db_context() as db:
            now = utcnow()
            sessions = (await db.scalars(select(CalibrationCaptureSession).where(or_(
                CalibrationCaptureSession.status.in_(ACTIVE_STATUSES),
                CalibrationCaptureSession.compute_status.in_(("queued", "running")),
            )).with_for_update(skip_locked=True))).all()
            for session in sessions:
                if interrupted and session.status == "running":
                    finish(session, "failed", "runtime_interrupted", now)
                    session.error_code = "capture_interrupted"
                else:
                    expire_session(session, now, self.config)
                if interrupted and session.compute_status == "running":
                    fail_compute(session, "calibration_interrupted")
                else:
                    expire_computation(session, self.config)
            for status, code in (("pending", "prediction_queue_timeout"), ("processing", "prediction_timeout")):
                timeout = self.config.CAPTURE_QUEUE_TIMEOUT_SEC if status == "pending" else self.config.PREDICTION_TIMEOUT_SEC
                stamp = PredictionJob.created_at if status == "pending" else PredictionJob.started_at
                condition = or_(PredictionJob.deadline_at <= now,
                                PredictionJob.deadline_at.is_(None) & (stamp <= now - timedelta(seconds=timeout)))
                if interrupted and status == "processing":
                    condition, code = True, "prediction_interrupted"
                result = await db.execute(update(PredictionJob).where(PredictionJob.status == status, condition).values(
                    status="failed", error_msg=code, result=None, completed_at=now, run_token=None,
                ))
                if result.rowcount:
                    logger.error(
                        "Runtime reconcile failed jobs status=%s code=%s interrupted=%s count=%d",
                        status, code, interrupted, result.rowcount,
                    )
            cameras = (await db.scalars(select(Camera).options(raiseload("*")).where(
                Camera.auto_predict_run_token.is_not(None),
                True if interrupted else or_(Camera.auto_predict_deadline_at.is_(None), Camera.auto_predict_deadline_at <= now),
            ).with_for_update(skip_locked=True))).all()
            for camera in cameras:
                reset_auto_schedule(camera)
                camera.auto_predict_last_run_status = "failed"
                camera.auto_predict_last_run_error = "prediction_interrupted" if interrupted else "prediction_timeout"
                camera.auto_predict_last_run_at = now
                logger.error(
                    "Runtime reconcile failed camera=%s error=%s interrupted=%s",
                    camera.id, camera.auto_predict_last_run_error, interrupted,
                )

    async def next_capture(self) -> UUID | None:
        async with self.db_context() as db:
            return await db.scalar(select(CalibrationCaptureSession.id).where(
                CalibrationCaptureSession.status == "queued",
            ).order_by(CalibrationCaptureSession.created_at, CalibrationCaptureSession.id).limit(1))

    async def next_work(self) -> tuple[str, UUID] | None:
        async with self.db_context() as db:
            for offset in range(len(WORK_KINDS)):
                index = (self.cursor + offset) % len(WORK_KINDS)
                kind = WORK_KINDS[index]
                if kind == "prediction":
                    query = select(PredictionJob.id).where(PredictionJob.status == "pending").order_by(PredictionJob.created_at, PredictionJob.id)
                elif kind == "calibration":
                    query = select(CalibrationCaptureSession.id).where(
                        CalibrationCaptureSession.compute_status == "queued", CalibrationCaptureSession.status.not_in(ACTIVE_STATUSES),
                    ).order_by(CalibrationCaptureSession.compute_requested_at, CalibrationCaptureSession.id)
                else:
                    query = select(Camera.id).where(
                        Camera.stream_enabled.is_(True), Camera.rtsp_url.is_not(None),
                        Camera.auto_predict_config["enabled"].as_string() == "true", Camera.auto_predict_run_token.is_(None),
                        or_(Camera.auto_predict_next_run.is_(None), Camera.auto_predict_next_run <= utcnow()),
                    ).order_by(Camera.auto_predict_next_run.asc().nullsfirst(), Camera.id)
                identifier = await db.scalar(query.limit(1))
                if identifier is not None:
                    self.cursor = (index + 1) % len(WORK_KINDS)
                    return kind, identifier
        return None

    async def run(self, kind: str, identifier: UUID) -> None:
        try:
            runners = {"capture": CaptureRunner, "calibration": CalibrationComputeRunner,
                       "prediction": PredictionRunner, "automatic": AutoPredictionRunner}
            result = await runners[kind](identifier, self.db_context, self.storage, self.config).run()
            if kind == "automatic" and result is not None:
                await self.events.publish(identifier, result)
        except Exception as exc:
            logger.error("Runtime job failed kind=%s id=%s exception=%s", kind, identifier, type(exc).__name__)
        finally:
            self.notify()

    async def loop(self) -> None:
        try:
            while self.status == "running":
                self.wakeup.clear()
                # A reconnect would lose the session lock, so it must stop this runtime.
                if self.connection.invalidated or await self.connection.scalar(text("SELECT pg_backend_pid()")) != self.backend_pid:
                    raise RuntimeError("runtime_lock_lost")
                await self.reconcile()
                if self.capture_task is None or self.capture_task.done():
                    self.capture_task = None
                    identifier = await self.next_capture()
                    if identifier is not None:
                        self.capture_task = asyncio.create_task(self.run("capture", identifier), name="sivera-capture")
                if self.work_task is None or self.work_task.done():
                    self.work_task, self.work_kind = None, None
                    work = await self.next_work()
                    if work is not None:
                        self.work_kind = work[0]
                        self.work_task = asyncio.create_task(self.run(*work), name="sivera-vision-job")
                try:
                    await asyncio.wait_for(self.wakeup.wait(), self.config.RUNTIME_POLL_INTERVAL_SEC)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.status = "failed"
            logger.error("Local runtime stopped exception=%s", type(exc).__name__)
            await self.drain(0)
            self.events.close()

    async def drain(self, grace_sec: float) -> None:
        tasks = {task for task in (self.work_task, self.capture_task) if task is not None and not task.done()}
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=grace_sec)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        self.status = "stopping"
        if self.loop_task is not None:
            self.loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.loop_task
        await self.drain(self.config.RUNTIME_SHUTDOWN_GRACE_SEC)
        self.events.close()
        try:
            if self.connection is not None:
                await self.reconcile(interrupted=True)
        finally:
            await self.release_lock()
            self.status = "stopped"
