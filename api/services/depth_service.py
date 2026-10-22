"""Depth estimation service: wraps depth.py logic."""
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from api.core.config import settings
from api.core.storage import get_storage, generate_storage_key
from api.models.orm import Camera, Calibration, GroundModel, DepthScale
from api.services.image_service import encode_image

# Lazy model cache: loads each YOLO depth model once per process
_depth_model_cache: dict = {}


def predict_depth(
    image_bgr: np.ndarray,
    model_path: str,
    imgsz: int = 768,
    device: str = "cpu",
) -> np.ndarray:
    """Predict depth map from image using YOLO depth model."""
    from ultralytics import YOLO
    
    # Lazy-load and cache the model per process
    if model_path not in _depth_model_cache:
        _depth_model_cache[model_path] = YOLO(model_path)
    model = _depth_model_cache[model_path]
    
    results = model.predict(source=image_bgr, imgsz=imgsz, device=device, verbose=False)
    res = results[0]
    
    if getattr(res, "depth", None) is None:
        raise RuntimeError("Model does not produce depth output. Ensure it's a YOLO depth model.")
    
    depth = res.depth
    if hasattr(depth, "data"):
        depth = depth.data
    if hasattr(depth, "cpu"):
        depth = depth.cpu().numpy()
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    
    h, w = image_bgr.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return depth


def calibrate_scale(
    depth_raw: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    img_pts: np.ndarray,
    world_xy: np.ndarray,
    min_pts: int = 3,
) -> Tuple[float, float, np.ndarray]:
    """Calibrate depth scale: depth_meter = s * depth_raw + b."""
    img_pts = np.asarray(img_pts, np.float64)
    world_xy = np.asarray(world_xy, np.float64)
    world_pts = np.hstack([world_xy, np.zeros((len(world_xy), 1))])  # Z=0
    
    # Ground truth depth (Z_camera) from pose
    cam = (R @ world_pts.T).T + t.ravel()
    d_gt = cam[:, 2]
    
    h, w = depth_raw.shape[:2]
    d_raw, valid_gt = [], []
    
    for (u, v), dg in zip(img_pts, d_gt):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < h and 0 <= ui < w and dg > 0:
            dr = float(depth_raw[vi, ui])
            if dr > 0 and np.isfinite(dr):
                d_raw.append(dr)
                valid_gt.append(dg)
    
    if len(valid_gt) < min_pts:
        raise RuntimeError(f"Only {len(valid_gt)} valid points for scale calibration (min {min_pts})")
    
    d_raw = np.array(d_raw, np.float64)
    d_gt = np.array(valid_gt, np.float64)
    
    # Least squares: d_gt = s * d_raw + b
    A = np.stack([d_raw, np.ones_like(d_raw)], axis=1)
    (s, b), *_ = np.linalg.lstsq(A, d_gt, rcond=None)
    residuals = (s * d_raw + b) - d_gt
    
    return float(s), float(b), residuals


def apply_scale(
    depth_raw: np.ndarray,
    s: float,
    b: float,
    clip_min: float = 0.1,
    clip_max: float = 50.0,
) -> np.ndarray:
    """Apply linear scale and clip to reasonable range."""
    depth_m = s * depth_raw.astype(np.float32) + b
    return np.clip(depth_m, clip_min, clip_max)


def lift_to_ground(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    depth_m: np.ndarray,
    waste_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lift waste pixels to 3D ground coordinates."""
    v_idx, u_idx = np.nonzero(waste_mask)
    z = depth_m[v_idx, u_idx]
    
    # Reconstruct 3D in camera coordinates
    P_cam = np.stack([
        (u_idx - K[0, 2]) / K[0, 0] * z,
        (v_idx - K[1, 2]) / K[1, 1] * z,
        z
    ], axis=1)
    
    # Transform to world coordinates
    cam_pos = -R.T @ t.ravel()
    P_world = (R.T @ P_cam.T).T + cam_pos
    
    # Height = Z_world (ground is Z=0)
    height = np.clip(P_world[:, 2], 0.0, None)
    X = P_world[:, 0]
    Y = P_world[:, 1]
    
    return X, Y, height, np.stack([u_idx, v_idx], axis=1)


def depth_colormap(
    depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Render depth map as color image."""
    d = depth.copy().astype(np.float32)
    if mask is not None:
        roi = d[mask > 0]
        if roi.size == 0:
            return np.zeros((*depth.shape, 3), np.uint8)
        lo, hi = np.percentile(roi, [2, 98])
    else:
        lo, hi = np.percentile(d, [2, 98])
    
    if hi <= lo:
        hi = lo + 1e-6
    d_norm = np.clip((d - lo) / (hi - lo), 0, 1)
    d_u8 = (d_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_u8, colormap)
    
    if mask is not None:
        m = (mask > 0).astype(np.uint8)
        colored = cv2.bitwise_and(colored, colored, mask=m)
    
    return colored


def height_heatmap(
    X: np.ndarray, Y: np.ndarray, height: np.ndarray, uv: np.ndarray,
    img_shape: Tuple[int, int], colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Render height map as color image."""
    out = np.zeros((*img_shape[:2], 3), np.uint8)
    if len(height) == 0:
        return out
    
    lo, hi = 0.0, np.percentile(height, 98)
    if hi <= lo:
        hi = lo + 0.01
    norm_h = np.clip((height - lo) / (hi - lo), 0, 1)
    colors = cv2.applyColorMap(
        (norm_h * 255).astype(np.uint8).reshape(-1, 1, 1),
        colormap
    ).reshape(-1, 3)
    
    for (u, v), col in zip(uv, colors):
        out[v, u] = col
    return out


async def save_depth_scale(
    db: AsyncSession,
    camera: Camera,
    ground_model,
    depth_model: str,
    s: float,
    b: float,
    residuals: np.ndarray,
    clip_min: float = 0.1,
    clip_max: float = 50.0,
) -> DepthScale:
    """Save depth scale calibration to database."""
    # Deactivate previous scales for this camera/ground_model
    await db.execute(
        update(DepthScale)
        .where(DepthScale.camera_id == camera.id, DepthScale.ground_model_id == ground_model.id)
        .values(is_active=False)  # Need to add is_active column
    )
    
    depth_scale = DepthScale(
        camera_id=camera.id,
        ground_model_id=ground_model.id,
        depth_model=depth_model,
        scale_s=s,
        offset_b=b,
        residuals=residuals.tolist(),
        mean_residual_m=float(np.abs(residuals).mean()),
        max_residual_m=float(np.abs(residuals).max()),
        clip_min_m=clip_min,
        clip_max_m=clip_max,
    )
    db.add(depth_scale)
    await db.flush()
    return depth_scale


async def run_depth_pipeline(
    db: AsyncSession,
    camera: Camera,
    calibration: Calibration,
    ground_model: GroundModel,
    depth_scale: DepthScale,
    image: np.ndarray,
    mask: np.ndarray,
    job_id: str,
) -> dict:
    """Run complete depth pipeline: predict -> scale -> lift -> visualize."""
    storage = get_storage()
    
    # 1. Predict depth
    depth_model_path = Path(settings.MODELS_DIR) / depth_scale.depth_model
    if not depth_model_path.exists():
        raise FileNotFoundError(f"Depth model not found: {depth_scale.depth_model} (looked in {settings.MODELS_DIR})")
    
    depth_raw = predict_depth(
        image, str(depth_model_path),
        imgsz=settings.DEFAULT_DEPTH_IMGSZ,
        device="cpu"
    )
    if depth_raw.shape != image.shape[:2] or not np.isfinite(depth_raw).all():
        raise ValueError("prediction_invalid_depth")
    
    # Save depth raw visualization
    depth_raw_vis = depth_colormap(depth_raw)
    raw_key = generate_storage_key(str(camera.id), job_id, "depth_raw_vis", "jpg")
    import io
    buf = io.BytesIO(encode_image(depth_raw_vis))
    storage.save(raw_key, buf, "image/jpeg")
    
    # 2. Apply scale
    depth_m = apply_scale(
        depth_raw, depth_scale.scale_s, depth_scale.offset_b,
        depth_scale.clip_min_m, depth_scale.clip_max_m
    )
    
    # Save depth scaled visualization
    depth_scaled_vis = depth_colormap(depth_m)
    scaled_key = generate_storage_key(str(camera.id), job_id, "depth_scaled_vis", "jpg")
    buf = io.BytesIO(encode_image(depth_scaled_vis))
    storage.save(scaled_key, buf, "image/jpeg")
    
    # 3. Lift to ground
    K = np.array(calibration.k_matrix, np.float64)
    R = np.array(ground_model.R_matrix, np.float64)
    t = np.array(ground_model.t_vector, np.float64).reshape(3, 1)
    
    X, Y, H, uv = lift_to_ground(K, R, t, depth_m, mask)
    
    # Save lifted points
    lifted_key = generate_storage_key(str(camera.id), job_id, "lifted_points_npz", "npz")
    buf = io.BytesIO()
    np.savez(buf, X=X, Y=Y, height=H, uv=uv)
    buf.seek(0)
    storage.save(lifted_key, buf, "application/octet-stream")
    
    # 4. Height map visualization
    hmap = height_heatmap(X, Y, H, uv, image.shape)
    overlay = cv2.addWeighted(image, 0.6, hmap, 0.7, 0)
    hmap_key = generate_storage_key(str(camera.id), job_id, "height_map", "jpg")
    buf = io.BytesIO(encode_image(overlay))
    storage.save(hmap_key, buf, "image/jpeg")
    
    # 5. Depth masked visualization
    depth_masked_vis = depth_colormap(depth_m, mask)
    masked_key = generate_storage_key(str(camera.id), job_id, "depth_masked", "jpg")
    buf = io.BytesIO(encode_image(depth_masked_vis))
    storage.save(masked_key, buf, "image/jpeg")
    
    return {
        "depth_raw_key": raw_key,
        "depth_scaled_key": scaled_key,
        "lifted_points_key": lifted_key,
        "height_map_key": hmap_key,
        "depth_masked_key": masked_key,
        "n_points": len(X),
        "height_stats": {
            "min": float(H.min()),
            "max": float(H.max()),
            "mean": float(H.mean()),
            "median": float(np.median(H)),
            "std": float(H.std()),
        } if len(H) else None,
        "footprint": {
            "x_min": float(X.min()), "x_max": float(X.max()),
            "y_min": float(Y.min()), "y_max": float(Y.max()),
        } if len(X) else None,
    }
