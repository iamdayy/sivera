import asyncio
import tempfile
from pathlib import Path
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import Settings, settings
from api.core.storage import StorageBackend, get_storage
from api.models.orm import Calibration, Camera, GroundModel, DepthScale
from api.schemas.ground_ref import DepthScaleCreate, DepthMeasurement
from api.services.image_service import decode_prediction_image, encode_image
from api.services.vision_process_service import VisionError, run_vision


_inference_lock = asyncio.Lock()


def test_image_key(camera_id: UUID, ground_model_id: UUID, image_id: UUID, preview: bool = False) -> str:
    filename = "undistorted.png" if preview else "source.image"
    return f"cameras/{camera_id}/depth-scale-images/{ground_model_id}/{image_id}/{filename}"


def is_test_image_key(key: str, camera_id: UUID, ground_model_id: UUID) -> bool:
    parts = key.split("/")
    if len(parts) != 6 or parts[:3] != ["cameras", str(camera_id), "depth-scale-images"] or parts[3] != str(ground_model_id) or parts[5] != "source.image":
        return False
    try:
        return str(UUID(parts[4])) == parts[4]
    except ValueError:
        return False


def read_image(key: str, storage: StorageBackend, config: Settings = settings) -> np.ndarray:
    stream = storage.get(key)
    if stream is None:
        raise ValueError("depth_scale_image_not_found")
    try:
        content = stream.read(config.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
    finally:
        stream.close()
    return decode_prediction_image(content, config)


def prepare_image(image: np.ndarray, calibration: Calibration, camera: Camera) -> np.ndarray:
    # Keep the exact raw/legacy input contract used by prediction, without a module cycle.
    from api.services.prediction_orchestrator import prepare_prediction_image
    return prepare_prediction_image(image, calibration, camera, "upload")


def fit_depth_scale(depth: np.ndarray, body: DepthScaleCreate, config: Settings = settings) -> dict:
    if depth.ndim != 2 or not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError("depth_scale_invalid_depth")
    if any(point.x >= depth.shape[1] or point.y >= depth.shape[0] for point in body.known_distances):
        raise ValueError("depth_scale_pixel_out_of_bounds")
    raw = np.array([depth[point.y, point.x] for point in body.known_distances], np.float64)
    truth = np.array([point.distance_m for point in body.known_distances], np.float64)
    # A scale and offset cannot be identified from effectively constant depths.
    if np.ptp(raw) <= max(1e-6, float(np.max(np.abs(raw))) * 1e-3) or np.ptp(truth) <= 1e-3:
        raise ValueError("depth_scale_degenerate_fit")
    normalized = (raw - raw.mean()) / raw.std()
    design = np.column_stack([normalized, np.ones_like(normalized)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, truth, rcond=None)
    scale = float(coefficients[0] / raw.std())
    offset = float(coefficients[1] - scale * raw.mean())
    if rank != 2 or not np.isfinite([scale, offset]).all() or scale <= 0:
        raise ValueError("depth_scale_degenerate_fit")
    fitted = scale * raw + offset
    residuals = fitted - truth
    absolute = np.abs(residuals)
    if (fitted <= 0).any() or absolute.mean() > config.DEPTH_SCALE_MAX_MEAN_RESIDUAL_M or absolute.max() > config.DEPTH_SCALE_MAX_RESIDUAL_M or (absolute / truth).max() > config.DEPTH_SCALE_MAX_RELATIVE_RESIDUAL:
        raise ValueError("depth_scale_poor_fit")
    return {"depth_model": body.depth_model, "scale_s": scale, "offset_b": offset,
            "residuals": residuals.tolist(), "mean_residual_m": float(absolute.mean()),
            "max_residual_m": float(absolute.max()), "clip_min_m": body.clip_min_m, "clip_max_m": body.clip_max_m}


def validate_saved_scale(scale: DepthScale, config: Settings = settings) -> None:
    values = [scale.scale_s, scale.offset_b, scale.clip_min_m, scale.clip_max_m,
              scale.mean_residual_m, scale.max_residual_m]
    residuals = np.asarray(scale.residuals, np.float64)
    if (any(value is None for value in values) or not np.isfinite(values).all()
            or scale.scale_s <= 0 or not 0 < scale.clip_min_m < scale.clip_max_m
            or residuals.ndim != 1 or len(residuals) < 3 or not np.isfinite(residuals).all()
            or not np.isclose(np.abs(residuals).mean(), scale.mean_residual_m)
            or not np.isclose(np.abs(residuals).max(), scale.max_residual_m)
            or scale.mean_residual_m > config.DEPTH_SCALE_MAX_MEAN_RESIDUAL_M
            or scale.max_residual_m > config.DEPTH_SCALE_MAX_RESIDUAL_M):
        raise ValueError("depth_scale_invalid_saved_fit")


async def compute_depth_scale(
    db: AsyncSession, camera: Camera, calibration: Calibration, ground_model: GroundModel,
    test_image_key: str, depth_model: str, known_distances: list[DepthMeasurement],
    clip_min_m: float = 0.1, clip_max_m: float = 50.0,
) -> dict:
    from api.services.depth_service import resolve_depth_model
    body = DepthScaleCreate(test_image=test_image_key, depth_model=depth_model,
                            known_distances=known_distances, clip_min_m=clip_min_m, clip_max_m=clip_max_m)
    model = resolve_depth_model(body.depth_model)
    image = prepare_image(read_image(body.test_image, get_storage()), calibration, camera)
    if any(point.x >= image.shape[1] or point.y >= image.shape[0] for point in body.known_distances):
        raise ValueError("depth_scale_pixel_out_of_bounds")
    if _inference_lock.locked():
        raise VisionError("depth_scale_busy")
    async with _inference_lock:
        with tempfile.TemporaryDirectory(prefix="sivera-depth-scale-") as directory:
            filename = Path(directory) / "image.png"
            filename.write_bytes(encode_image(image, ".png"))
            payload = {"image_path": str(filename), "model_path": str(model), "imgsz": settings.DEFAULT_DEPTH_IMGSZ,
                       "body": body.model_dump(), "max_mean": settings.DEPTH_SCALE_MAX_MEAN_RESIDUAL_M,
                       "max_residual": settings.DEPTH_SCALE_MAX_RESIDUAL_M,
                       "max_relative": settings.DEPTH_SCALE_MAX_RELATIVE_RESIDUAL}
            try:
                return await run_vision("depth-scale", payload, settings.DEPTH_SCALE_TIMEOUT_SEC)
            except VisionError as exc:
                if exc.code in {"depth_scale_invalid_depth", "depth_scale_pixel_out_of_bounds", "depth_scale_degenerate_fit", "depth_scale_poor_fit"}:
                    raise ValueError(exc.code) from None
                raise
