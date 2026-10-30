import asyncio
import hashlib
import io
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
from api.services.depth_model_service import depth_model_identity
from api.services.depth_service import depth_colormap, predict_depth


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


def depth_preview_key(camera_id: UUID, ground_model_id: UUID, depth_image_id: UUID, preview: bool = False) -> str:
    """Storage key for temporary depth preview data (TTL ~10 min)."""
    filename = "depth_vis.png" if preview else "depth_raw.npy"
    return f"cameras/{camera_id}/depth-previews/{ground_model_id}/{depth_image_id}/{filename}"


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
    if getattr(scale, "model_identity", None):
        provenance = scale.fit_provenance
        try:
            body = DepthScaleCreate(known_distances=provenance["known_distances"],
                clip_min_m=scale.clip_min_m, clip_max_m=scale.clip_max_m, depth_model=scale.depth_model)
            truth = np.array([point.distance_m for point in body.known_distances])
            if len(truth) != len(residuals) or (np.abs(residuals) / truth).max() > config.DEPTH_SCALE_MAX_RELATIVE_RESIDUAL:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise ValueError("depth_scale_invalid_saved_fit") from None


async def compute_depth_preview(
    db: AsyncSession, camera: Camera, calibration: Calibration, ground_model: GroundModel,
    test_image_key: str, depth_model: str,
) -> dict:
    """Run YOLO depth inference on test image and return depth visualization + stats.
    
    Stores raw depth array temporarily for the subsequent create step.
    Returns: depth_image_id, depth_raw, depth_vis, depth_stats, width, height, model_used, imgsz
    """
    from api.services.depth_service import resolve_depth_model
    
    model = resolve_depth_model(depth_model)
    identity = await asyncio.to_thread(depth_model_identity, model)
    image = prepare_image(read_image(test_image_key, get_storage()), calibration, camera)
    
    if _inference_lock.locked():
        raise VisionError("depth_scale_busy")
    
    async with _inference_lock:
        with tempfile.TemporaryDirectory(prefix="sivera-depth-preview-") as directory:
            filename = Path(directory) / "image.png"
            prepared = encode_image(image, ".png")
            filename.write_bytes(prepared)
            
            try:
                result = await run_vision("depth-preview", {
                    "image_path": str(filename),
                    "model_path": str(model),
                    "imgsz": settings.DEFAULT_DEPTH_IMGSZ,
                }, settings.DEPTH_SCALE_TIMEOUT_SEC)
                
                # Verify model identity hasn't changed
                if result.get("model_identity") != identity or await asyncio.to_thread(depth_model_identity, model) != identity:
                    raise ValueError("depth_scale_model_mismatch")
                
                # Extract depth raw from result (base64 encoded)
                import base64
                depth_raw = np.frombuffer(base64.b64decode(result["depth_raw"]), dtype=np.float32).reshape(result["height"], result["width"])
                depth_vis_b64 = result["visualization"]
                depth_vis = decode_prediction_image(base64.b64decode(depth_vis_b64))
                
                # Generate depth_image_id
                depth_image_id = UUID(result["depth_image_id"])
                
                # Store raw depth and visualization in temporary storage
                storage = get_storage()
                raw_buf = io.BytesIO()
                np.save(raw_buf, depth_raw)
                raw_buf.seek(0)
                storage.save(depth_preview_key(camera.id, ground_model.id, depth_image_id, False), raw_buf, "application/octet-stream")
                
                vis_buf = io.BytesIO(encode_image(depth_vis, ".png"))
                storage.save(depth_preview_key(camera.id, ground_model.id, depth_image_id, True), vis_buf, "image/png")
                
                return {
                    "depth_image_id": depth_image_id,
                    "width": result["width"],
                    "height": result["height"],
                    "depth_stats": result["depth_stats"],
                    "model_used": depth_model,
                    "imgsz": settings.DEFAULT_DEPTH_IMGSZ,
                }
            except VisionError as exc:
                if exc.code in {"depth_scale_invalid_depth", "depth_scale_pixel_out_of_bounds", "depth_scale_degenerate_fit", "depth_scale_poor_fit", "depth_scale_model_mismatch", "depth_model_unavailable"}:
                    raise ValueError(exc.code) from None
                raise


async def compute_depth_scale(
    db: AsyncSession, camera: Camera, calibration: Calibration, ground_model: GroundModel,
    test_image_key: str, depth_model: str, known_distances: list[DepthMeasurement],
    clip_min_m: float = 0.1, clip_max_m: float = 50.0,
) -> dict:
    from api.services.depth_service import resolve_depth_model
    body = DepthScaleCreate(known_distances=known_distances, depth_model=depth_model,
                           clip_min_m=clip_min_m, clip_max_m=clip_max_m)
    model = resolve_depth_model(depth_model)
    identity = await asyncio.to_thread(depth_model_identity, model)
    image = prepare_image(read_image(test_image_key, get_storage()), calibration, camera)
    if any(point.x >= image.shape[1] or point.y >= image.shape[0] for point in known_distances):
        raise ValueError("depth_scale_pixel_out_of_bounds")
    if _inference_lock.locked():
        raise VisionError("depth_scale_busy")
    async with _inference_lock:
        with tempfile.TemporaryDirectory(prefix="sivera-depth-scale-") as directory:
            filename = Path(directory) / "image.png"
            prepared = encode_image(image, ".png")
            filename.write_bytes(prepared)
            payload = {"image_path": str(filename), "model_path": str(model), "imgsz": settings.DEFAULT_DEPTH_IMGSZ,
                       "body": body.model_dump(), "max_mean": settings.DEPTH_SCALE_MAX_MEAN_RESIDUAL_M,
                       "max_residual": settings.DEPTH_SCALE_MAX_RESIDUAL_M,
                       "max_relative": settings.DEPTH_SCALE_MAX_RELATIVE_RESIDUAL}
            try:
                result = await run_vision("depth-scale", payload, settings.DEPTH_SCALE_TIMEOUT_SEC)
                if result.get("model_identity") != identity or await asyncio.to_thread(depth_model_identity, model) != identity:
                    raise ValueError("depth_scale_model_mismatch")
                result["fit_provenance"] = {
                    "source_key": test_image_key,
                    "prepared_image_sha256": hashlib.sha256(prepared).hexdigest(),
                    "known_distances": [point.model_dump() for point in known_distances],
                    "max_mean_residual_m": settings.DEPTH_SCALE_MAX_MEAN_RESIDUAL_M,
                    "max_residual_m": settings.DEPTH_SCALE_MAX_RESIDUAL_M,
                    "max_relative_residual": settings.DEPTH_SCALE_MAX_RELATIVE_RESIDUAL,
                }
                return result
            except VisionError as exc:
                if exc.code in {"depth_scale_invalid_depth", "depth_scale_pixel_out_of_bounds", "depth_scale_degenerate_fit", "depth_scale_poor_fit", "depth_scale_model_mismatch", "depth_model_unavailable"}:
                    raise ValueError(exc.code) from None
                raise


async def compute_depth_scale_from_preview(
    db: AsyncSession, camera: Camera, calibration: Calibration, ground_model: GroundModel,
    depth_image_id: UUID, depth_model: str, known_distances: list[DepthMeasurement],
    clip_min_m: float = 0.1, clip_max_m: float = 50.0,
) -> dict:
    """Fit depth scale using stored raw depth from preview step.
    
    Retrieves raw depth from temporary storage, fits scale, cleans up temp storage.
    """
    from api.services.depth_service import resolve_depth_model
    from api.core.storage import get_storage
    
    body = DepthScaleCreate(known_distances=known_distances, depth_model=depth_model,
                           clip_min_m=clip_min_m, clip_max_m=clip_max_m)
    model = resolve_depth_model(depth_model)
    identity = await asyncio.to_thread(depth_model_identity, model)
    
    # Retrieve raw depth from temporary storage
    storage = get_storage()
    raw_buf = io.BytesIO()
    stream = storage.get(depth_preview_key(camera.id, ground_model.id, depth_image_id, False))
    if stream is None:
        raise ValueError("depth_preview_expired")
    try:
        content = stream.read(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
    finally:
        stream.close()
    raw_buf.write(content)
    raw_buf.seek(0)
    depth_raw = np.load(raw_buf)
    
    # Verify dimensions match
    if any(point.x >= depth_raw.shape[1] or point.y >= depth_raw.shape[0] for point in known_distances):
        raise ValueError("depth_scale_pixel_out_of_bounds")
    
    # Verify model identity
    current_identity = await asyncio.to_thread(depth_model_identity, model)
    if current_identity != identity:
        raise ValueError("depth_scale_model_mismatch")
    
    # Fit scale using stored raw depth
    fit_result = fit_depth_scale(depth_raw, body)
    
    # Clean up temporary storage
    try:
        storage.delete(depth_preview_key(camera.id, ground_model.id, depth_image_id, False))
        storage.delete(depth_preview_key(camera.id, ground_model.id, depth_image_id, True))
    except Exception:
        pass  # Best effort cleanup
    
    fit_result["model_identity"] = identity
    fit_result["fit_provenance"] = {
        "depth_image_id": str(depth_image_id),
        "known_distances": [point.model_dump() for point in known_distances],
        "max_mean_residual_m": settings.DEPTH_SCALE_MAX_MEAN_RESIDUAL_M,
        "max_residual_m": settings.DEPTH_SCALE_MAX_RESIDUAL_M,
        "max_relative_residual": settings.DEPTH_SCALE_MAX_RELATIVE_RESIDUAL,
    }
    return fit_result
