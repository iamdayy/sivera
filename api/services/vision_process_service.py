import asyncio
import json
import logging
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from weakref import WeakKeyDictionary

from api.core.config import settings

logger = logging.getLogger(__name__)


class VisionError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_slots: WeakKeyDictionary = WeakKeyDictionary()
_owner: ContextVar[asyncio.Task | None] = ContextVar("vision_owner", default=None)


@asynccontextmanager
async def vision_slot(timeout_sec: float, heartbeat: Callable[[], Awaitable[bool]] | None = None):
    if _owner.get() is asyncio.current_task():
        yield
        return
    slot = _slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(1))
    acquired = False
    try:
        async with asyncio.timeout(timeout_sec):
            while not acquired:
                if heartbeat is not None and not await heartbeat():
                    raise VisionError("vision_cancelled")
                try:
                    await asyncio.wait_for(slot.acquire(), timeout=1)
                    acquired = True
                except TimeoutError:
                    continue
        token = _owner.set(asyncio.current_task())
        try:
            yield
        finally:
            _owner.reset(token)
    except TimeoutError:
        raise VisionError("vision_timeout") from None
    finally:
        if acquired:
            slot.release()


async def _exchange(process: asyncio.subprocess.Process, content: bytes, limit: int) -> bytes:
    async def write() -> None:
        try:
            process.stdin.write(content)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    async def read() -> bytes:
        output = bytearray()
        while chunk := await process.stdout.read(65536):
            output.extend(chunk)
            if len(output) > limit:
                raise VisionError("vision_message_too_large")
        return bytes(output)

    writer = asyncio.create_task(write())
    try:
        output = await read()
        await writer
        await process.wait()
        return output
    finally:
        if not writer.done():
            writer.cancel()
        with suppress(asyncio.CancelledError):
            await writer


async def run_vision(
    mode: str, payload: dict, timeout_sec: float,
    heartbeat: Callable[[], Awaitable[bool]] | None = None,
) -> dict:
    loop = asyncio.get_running_loop()
    content = await asyncio.to_thread(lambda: json.dumps(payload, allow_nan=False).encode())
    if len(content) > settings.VISION_MAX_MESSAGE_BYTES:
        logger.error("Vision error mode=%s message_bytes=%d limit=%d code=vision_message_too_large", mode, len(content), settings.VISION_MAX_MESSAGE_BYTES)
        raise VisionError("vision_message_too_large")
    process, task = None, None
    started = loop.time()
    try:
        async with asyncio.timeout(timeout_sec), vision_slot(timeout_sec, heartbeat):
            if heartbeat is not None and not await heartbeat():
                logger.error("Vision cancelled mode=%s heartbeat=False", mode)
                raise VisionError("vision_cancelled")
            env = os.environ.copy()
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = str(settings.VISION_CPU_THREADS)
            # The external timer also bounds children after an abrupt API process death.
            spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                "timeout", "--signal=KILL", str(max(0.1, timeout_sec - (loop.time() - started))),
                sys.executable, "-m", "api.runtime.vision_process", mode,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, start_new_session=True, env=env,
            ))
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                process = await spawn
                raise
            task = asyncio.create_task(_exchange(process, content, settings.VISION_MAX_MESSAGE_BYTES))
            while not task.done():
                await asyncio.wait({task}, timeout=1)
                if heartbeat is not None and not await heartbeat():
                    logger.error("Vision cancelled mode=%s heartbeat=False", mode)
                    raise VisionError("vision_cancelled")
            output = task.result()
            if process.returncode in {124, 137, -signal.SIGKILL}:
                logger.error("Vision timeout mode=%s returncode=%s", mode, process.returncode)
                raise VisionError("vision_timeout")
            if process.returncode != 0:
                logger.error("Vision failed mode=%s returncode=%s", mode, process.returncode)
                raise VisionError("vision_failed")
            result = await asyncio.to_thread(json.loads, output)
            if not isinstance(result, dict):
                logger.error("Vision failed mode=%s result_not_dict", mode)
                raise VisionError("vision_failed")
            if "error" in result:
                logger.error("Vision failed mode=%s subprocess_error=%s", mode, result["error"])
                raise VisionError(result["error"])
            return result
    except TimeoutError:
        logger.error("Vision timeout mode=%s timeout_sec=%s", mode, timeout_sec)
        raise VisionError("vision_timeout") from None
    except OSError:
        logger.error("Vision runtime missing mode=%s", mode)
        raise VisionError("vision_runtime_missing") from None
    except (ValueError, TypeError):
        logger.error("Vision failed mode=%s exception=value_or_type_error", mode)
        raise VisionError("vision_failed") from None
    finally:
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, VisionError):
                await task
