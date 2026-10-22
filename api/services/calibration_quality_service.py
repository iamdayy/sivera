from dataclasses import dataclass
import base64
from collections.abc import Awaitable, Callable

import cv2
import numpy as np

from api.core.config import Settings, settings
from api.services.rtsp_capture_service import CapturedFrame
from api.services.vision_process_service import run_vision


@dataclass(frozen=True)
class FrameQuality:
    status: str
    reason: str | None
    sharpness: float
    corners: list[list[float]] | None
    descriptor: list[float] | None


async def assess_frame_bounded(
    frame: CapturedFrame, cols: int, rows: int, previous: list[list[float]],
    config: Settings = settings, heartbeat: Callable[[], Awaitable[bool]] | None = None,
) -> FrameQuality:
    result = await run_vision("assess", {
        "image": base64.b64encode(frame.content).decode(), "cols": cols, "rows": rows,
        "previous": previous, "min_sharpness": config.CALIB_CAPTURE_MIN_SHARPNESS,
        "max_frame_bytes": config.CAPTURE_MAX_FRAME_BYTES, "max_pixels": config.CAPTURE_MAX_PIXELS,
    }, config.STREAM_CAPTURE_TIMEOUT, heartbeat)
    return FrameQuality(**result)


def pose_descriptor(corners: np.ndarray, width: int, height: int, cols: int, rows: int) -> list[float]:
    grid = corners.reshape(rows, cols, 2).astype(np.float64)
    normalized = grid / [width, height]
    center = normalized.mean(axis=(0, 1))
    horizontal = normalized[0, -1] - normalized[0, 0]
    vertical = normalized[-1, 0] - normalized[0, 0]
    top, bottom = np.linalg.norm(horizontal), np.linalg.norm(normalized[-1, -1] - normalized[-1, 0])
    left, right = np.linalg.norm(vertical), np.linalg.norm(normalized[-1, -1] - normalized[0, -1])
    angle = np.arctan2(horizontal[1], horizontal[0]) % np.pi
    return [float(center[0]), float(center[1]), float(top), float(left), float(angle),
            float(bottom / max(top, 1e-9)), float(right / max(left, 1e-9))]


def similar_pose(candidate: list[float], previous: list[float]) -> bool:
    angle = abs(candidate[4] - previous[4])
    angle = min(angle, np.pi - angle)
    return (
        np.linalg.norm(np.array(candidate[:2]) - previous[:2]) < 0.07
        and abs(np.log(max(candidate[2], 1e-9) / max(previous[2], 1e-9))) < 0.15
        and abs(np.log(max(candidate[3], 1e-9) / max(previous[3], 1e-9))) < 0.15
        and angle < np.deg2rad(10)
        and max(abs(candidate[5] - previous[5]), abs(candidate[6] - previous[6])) < 0.15
    )


def assess_frame(frame: CapturedFrame, cols: int, rows: int, previous: list[list[float]], config: Settings = settings) -> FrameQuality:
    image = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return FrameQuality("rejected", "invalid_image", 0, None, None)
    # Corner coordinates always refer to the original image, never a resized detector input.
    found, corners = detect_chessboard(image, cols, rows)
    if not found:
        return FrameQuality("rejected", "chessboard_not_found", float(cv2.Laplacian(image, cv2.CV_64F).var()), None, None)
    points = corners.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    roi = image[max(0, y):min(frame.height, y + h), max(0, x):min(frame.width, x + w)]
    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0
    descriptor = pose_descriptor(points, frame.width, frame.height, cols, rows)
    reason = "blur" if sharpness < config.CALIB_CAPTURE_MIN_SHARPNESS else None
    if reason is None and any(similar_pose(descriptor, old) for old in previous):
        reason = "similar_pose"
    return FrameQuality("rejected" if reason else "accepted", reason, sharpness, points.tolist(), descriptor)


def detect_chessboard(
    image: np.ndarray, cols: int, rows: int, flags: int = cv2.CALIB_CB_NORMALIZE_IMAGE,
) -> tuple[bool, np.ndarray | None]:
    if cols < 3 or rows < 3:
        raise ValueError("calibration_invalid_parameters")
    return cv2.findChessboardCornersSB(image, (cols, rows), flags=flags)


def corner_preview(content: bytes, corners: list[list[float]] | None, cols: int, rows: int) -> bytes:
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("capture_invalid_frame")
    if corners:
        cv2.drawChessboardCorners(image, (cols, rows), np.array(corners, np.float32).reshape(-1, 1, 2), True)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("capture_preview_failed")
    return encoded.tobytes()
