import asyncio
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress


class VisionError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


async def run_vision(
    mode: str, payload: dict, timeout_sec: float,
    heartbeat: Callable[[], Awaitable[bool]] | None = None,
) -> dict:
    content = json.dumps(payload, allow_nan=False).encode()
    # A subprocess can be killed; cancelling to_thread does not stop OpenCV computation.
    try:
        process = await asyncio.create_subprocess_exec(
            "timeout", "--signal=KILL", str(timeout_sec), sys.executable,
            "-m", "api.workers.vision_process", mode,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
    except OSError:
        raise VisionError("vision_runtime_missing") from None
    task = asyncio.create_task(process.communicate(content))
    try:
        async with asyncio.timeout(timeout_sec):
            while not task.done():
                await asyncio.wait({task}, timeout=1)
                if heartbeat is not None and not await heartbeat():
                    raise VisionError("vision_cancelled")
            output, _ = task.result()
        if process.returncode in {124, 137, -signal.SIGKILL}:
            raise VisionError("vision_timeout")
        if process.returncode != 0 or len(output) > 1024 * 1024:
            raise VisionError("vision_failed")
        result = json.loads(output)
        if "error" in result:
            raise VisionError(result["error"])
        return result
    except TimeoutError:
        raise VisionError("vision_timeout") from None
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
