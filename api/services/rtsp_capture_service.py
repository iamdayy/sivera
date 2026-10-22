import asyncio
import io
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote
from uuid import UUID

from PIL import Image, UnidentifiedImageError

from api.core.config import Settings, settings
from api.services.mediamtx_service import MediaMTXError, MediaMTXService, desired_path, stream_path


class CaptureError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CapturedFrame:
    content: bytes
    width: int
    height: int
    captured_at: datetime


def relay_url(camera_id: UUID, config: Settings = settings) -> str:
    if not config.MEDIAMTX_ENABLED or not config.MEDIAMTX_READ_USER or not config.MEDIAMTX_READ_PASSWORD.get_secret_value():
        raise CaptureError("stream_service_not_configured")
    host = config.MEDIAMTX_HOST
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    user = quote(config.MEDIAMTX_READ_USER, safe="")
    password = quote(config.MEDIAMTX_READ_PASSWORD.get_secret_value(), safe="")
    return f"rtsp://{user}:{password}@{host}:{config.MEDIAMTX_RTSP_PORT}/{stream_path(camera_id)}"


def decode_frame(content: bytes, config: Settings = settings) -> CapturedFrame:
    if len(content) > config.CAPTURE_MAX_FRAME_BYTES:
        raise CaptureError("capture_frame_too_large")
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > config.CAPTURE_MAX_PIXELS:
                raise CaptureError("capture_resolution_too_large")
            if image.format != "JPEG":
                raise CaptureError("capture_invalid_frame")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise CaptureError("capture_invalid_frame") from None
    return CapturedFrame(content, width, height, datetime.now(timezone.utc))


async def check_relay_source(camera_id: UUID, source: str, config: Settings = settings) -> None:
    relay = MediaMTXService(config)
    try:
        current = await relay.get_path_config(stream_path(camera_id))
        if current is None or any(current.get(key) != value for key, value in desired_path(source).items()):
            raise CaptureError("capture_stream_sync_pending")
    except MediaMTXError as exc:
        raise CaptureError(exc.code) from None
    finally:
        await relay.close()


async def capture_frame(camera_id: UUID, config: Settings = settings) -> CapturedFrame:
    url = relay_url(camera_id, config)
    try:
        process = await asyncio.create_subprocess_exec(
            # This timer outlives a killed Celery child; Python's timeout alone does not.
            "timeout", "--signal=KILL", str(config.STREAM_CAPTURE_TIMEOUT),
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-max_alloc", "134217728", "-rtsp_transport", "tcp",
            "-timeout", str(config.STREAM_CAPTURE_TIMEOUT * 1000000),
            "-max_pixels", str(config.CAPTURE_MAX_PIXELS),
            "-threads", "1", "-noautorotate", "-i", url, "-map", "0:v:0", "-an",
            "-frames:v", "1", "-threads", "1", "-c:v", "mjpeg", "-q:v", "2",
            "-f", "image2pipe", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise CaptureError("capture_runtime_missing") from None
    except OSError:
        raise CaptureError("capture_process_failed") from None
    try:
        async with asyncio.timeout(config.STREAM_CAPTURE_TIMEOUT):
            content = bytearray()
            while chunk := await process.stdout.read(65536):
                content.extend(chunk)
                if len(content) > config.CAPTURE_MAX_FRAME_BYTES:
                    raise CaptureError("capture_frame_too_large")
            returncode = await process.wait()
            if returncode in {124, 137, -signal.SIGKILL}:
                raise CaptureError("capture_timeout")
            if returncode == 127:
                raise CaptureError("capture_ffmpeg_missing")
            if returncode != 0:
                raise CaptureError("capture_stream_unavailable")
        return decode_frame(bytes(content), config)
    except TimeoutError:
        raise CaptureError("capture_timeout") from None
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
