"""Ground reference and ground model service: wraps ipm.py logic."""
import json
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
from api.models.orm import Camera, Calibration, GroundReference, GroundModel
from api.services.image_service import encode_image


def compute_ground_model(
    image_points: np.ndarray,
    world_points: np.ndarray,
    K: np.ndarray,
) -> dict:
    """
    Compute homography and camera pose from ground reference points.
    Equivalent to ipm.py ground_model function.
    """
    image_points = np.asarray(image_points, np.float64)
    world_points = np.asarray(world_points, np.float64)
    
    if len(image_points) < 4 or len(image_points) != len(world_points):
        raise ValueError("Need at least 4 corresponding points")
    
    # Homography world -> image
    H_w2i, _ = cv2.findHomography(world_points, image_points, 0)
    H_i2w = np.linalg.inv(H_w2i)
    
    # Camera pose via solvePnP (ground plane Z=0)
    obj_points = np.hstack([world_points, np.zeros((len(world_points), 1))])
    ok, rvec, tvec = cv2.solvePnP(
        obj_points, image_points, np.asarray(K, np.float64), None,
        flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_points, image_points, np.asarray(K, np.float64), None, rvec, tvec
    )
    R, _ = cv2.Rodrigues(rvec)
    
    # Reprojection error
    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, np.asarray(K, np.float64), None)
    err = np.linalg.norm(proj.reshape(-1, 2) - image_points, axis=1)
    
    # Camera position in world coordinates
    cam_pos = (-R.T @ tvec).ravel()
    cam_height = float(abs(cam_pos[2]))
    
    # Optical axis in world coordinates
    axis_w = R.T @ np.array([0.0, 0.0, 1.0])
    pitch_down = np.degrees(np.arcsin(np.clip(-axis_w[2] * np.sign(cam_pos[2]), -1, 1)))
    
    return {
        "H_w2i": H_w2i,
        "H_i2w": H_i2w,
        "R": R,
        "t": tvec.ravel(),
        "K": K,
        "cam_height": cam_height,
        "pitch_down_deg": float(pitch_down),
        "reproj_err": err,
        "reproj_err_mean": float(err.mean()),
        "reproj_err_max": float(err.max()),
    }


def compute_bev(
    image: np.ndarray,
    mask: np.ndarray,
    gm: dict,
    ppm: int = 200,
    margin_m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Compute BEV (Bird's Eye View) image and mask projection.
    Returns (bev_image, bev_mask_projection, bev_overlay, bev_info)
    """
    H_i2w = gm["H_i2w"]
    img_pts = np.array([[0, 0], [image.shape[1], 0], 
                        [image.shape[1], image.shape[0]], [0, image.shape[0]]], np.float64)
    world_pts, _ = to_world(H_i2w, img_pts)
    
    # Base contour from mask
    base_px, trunc = base_contour(mask)
    if len(base_px):
        base_w, ok = to_world(H_i2w, base_px)
        ok = ok & (np.abs(base_w) < 200).all(axis=1)
        base_w = base_w[ok]
        xs = np.r_[world_pts[:, 0], base_w[:, 0]]
        ys = np.r_[world_pts[:, 1], base_w[:, 1]]
    else:
        xs, ys = world_pts[:, 0], world_pts[:, 1]
    
    xmin, xmax = xs.min() - margin_m, xs.max() + margin_m
    ymin, ymax = ys.min() - margin_m, ys.max() + margin_m
    
    wpx = int(round((xmax - xmin) * ppm))
    hpx = int(round((ymax - ymin) * ppm))
    
    if max(wpx, hpx) > 6000:
        raise ValueError(f"BEV too large: {wpx}x{hpx}. Reduce ppm or margin.")
    
    Hb = bev_matrix(H_i2w, xmin, ymin, ppm)
    bev_img = cv2.warpPerspective(image, Hb, (wpx, hpx))
    bev_mask = cv2.warpPerspective(mask.astype(np.uint8) * 255, Hb, (wpx, hpx),
                                    flags=cv2.INTER_NEAREST)
    
    # Create overlay with grid and reference points
    overlay = bev_img.copy()
    draw_bev_overlay(overlay, gm, H_i2w, xmin, ymin, ppm, wpx, hpx, 
                     base_w if len(base_px) else None)
    
    bev_info = {
        "width_px": wpx,
        "height_px": hpx,
        "ppm": ppm,
        "x_min_m": float(xmin),
        "x_max_m": float(xmax),
        "y_min_m": float(ymin),
        "y_max_m": float(ymax),
        "margin_m": margin_m,
    }
    
    return bev_img, bev_mask, overlay, bev_info


def to_world(H_i2w: np.ndarray, px: np.ndarray, ref_sign: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel to world coordinates (meters)."""
    px = np.asarray(px, np.float64).reshape(-1, 2)
    p = np.hstack([px, np.ones((len(px), 1))]) @ H_i2w.T
    w = p[:, 2]
    xy = p[:, :2] / w[:, None]
    valid = np.isfinite(xy).all(axis=1)
    if ref_sign is not None:
        valid &= np.sign(w) == ref_sign
    return xy, valid


def ref_w_sign(H_i2w: np.ndarray, img_pts: np.ndarray) -> float:
    """Get reference w sign for valid points."""
    p = np.hstack([np.asarray(img_pts, np.float64), np.ones((len(img_pts), 1))]) @ H_i2w.T
    return np.sign(np.median(p[:, 2]))


def base_contour(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get bottom-most mask pixel per column (ground contact line)."""
    h = mask.shape[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, bool)
    v = h - 1 - np.argmax(mask[::-1, cols], axis=0)
    pts = np.stack([cols, v], axis=1).astype(np.float32)
    truncated = v >= h - 1
    return pts, truncated


def bev_matrix(H_i2w: np.ndarray, xmin: float, ymin: float, ppm: int) -> np.ndarray:
    """Build BEV transformation matrix."""
    S = np.array([[ppm, 0, -xmin * ppm], [0, ppm, -ymin * ppm], [0, 0, 1.0]])
    return S @ H_i2w


def draw_bev_overlay(
    overlay: np.ndarray,
    gm: dict,
    H_i2w: np.ndarray,
    xmin: float, ymin: float,
    ppm: int, wpx: int, hpx: int,
    base_w: Optional[np.ndarray] = None,
):
    """Draw grid, reference points, and base contour on BEV overlay."""
    # Grid lines every 1 meter
    to_bev = lambda xy: np.round((np.asarray(xy) - [xmin, ymin]) * ppm).astype(int)
    
    for gx in np.arange(np.ceil(xmin), xmin + wpx / ppm, 1.0):
        x = int(round((gx - xmin) * ppm))
        if 0 <= x < wpx:
            cv2.line(overlay, (x, 0), (x, hpx - 1), (255, 255, 255), 1)
            cv2.putText(overlay, f"{gx:g}m", (x + 3, 14), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    for gy in np.arange(np.ceil(ymin), ymin + hpx / ppm, 1.0):
        y = int(round((gy - ymin) * ppm))
        if 0 <= y < hpx:
            cv2.line(overlay, (0, y), (wpx - 1, y), (255, 255, 255), 1)
            cv2.putText(overlay, f"{gy:g}m", (3, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Reference points
    world_pts = np.array([[0, 0], [2, 0], [2, 3], [0, 3]], np.float64)  # Default 2x3 rect
    # Try to get actual world points from ground reference
    for wp in world_pts:
        pt = to_bev(wp)
        if 0 <= pt[0] < wpx and 0 <= pt[1] < hpx:
            cv2.circle(overlay, tuple(pt), 6, (0, 0, 255), -1)
    
    # Base contour
    if base_w is not None and len(base_w):
        poly = to_bev(base_w).reshape(-1, 1, 2)
        cv2.polylines(overlay, [poly], False, (0, 255, 255), 2)


async def save_ground_reference(
    db: AsyncSession,
    camera: Camera,
    calibration: Calibration,
    image_points: List[List[float]],
    world_points: List[List[float]],
    version: int,
    world_units: str = "meter",
    description: Optional[str] = None,
) -> GroundReference:
    """Save ground reference to database."""
    # Deactivate previous
    await db.execute(
        update(GroundReference)
        .where(GroundReference.camera_id == camera.id)
        .values(is_active=False)
    )
    
    gr = GroundReference(
        camera_id=camera.id,
        calibration_id=calibration.id,
        version=version,
        image_points=image_points,
        world_points=world_points,
        world_units=world_units,
        description=description,
        is_active=True,
    )
    db.add(gr)
    await db.flush()
    return gr


async def create_ground_model_from_ref(
    db: AsyncSession,
    camera: Camera,
    calibration: Calibration,
    ground_ref: GroundReference,
    ppm: int = 200,
    margin_m: float = 1.0,
) -> GroundModel:
    """Create ground model from ground reference."""
    K = np.array(calibration.k_matrix, np.float64)
    img_pts = np.array(ground_ref.image_points, np.float64)
    world_pts = np.array(ground_ref.world_points, np.float64)
    
    gm = compute_ground_model(img_pts, world_pts, K)
    
    ground_model = GroundModel(
        camera_id=camera.id,
        calibration_id=calibration.id,
        ground_ref_id=ground_ref.id,
        R_matrix=gm["R"].tolist(),
        t_vector=gm["t"].tolist(),
        H_i2w=gm["H_i2w"].tolist(),
        H_w2i=gm["H_w2i"].tolist(),
        cam_height=gm["cam_height"],
        pitch_down_deg=gm["pitch_down_deg"],
        reproj_err_px=gm["reproj_err_mean"],
        bev_ppm=ppm,
        bev_margin_m=margin_m,
    )
    db.add(ground_model)
    await db.flush()
    return ground_model


async def generate_bev_preview(
    db: AsyncSession,
    camera: Camera,
    calibration: Calibration,
    ground_model: GroundModel,
    test_image_path: str,
    mask_path: Optional[str] = None,
) -> dict:
    """Generate BEV preview images and save to storage."""
    storage = get_storage()
    
    # Load image from storage (test_image_path is a storage key)
    img_data = storage.get(test_image_path)
    if img_data is None:
        raise ValueError(f"Test image not found in storage: {test_image_path}")
    try:
        img_array = np.frombuffer(img_data.read(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1), np.uint8)
    finally:
        img_data.close()
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode test image: {test_image_path}")
    
    # Load or create mask
    if mask_path:
        mask_data = storage.get(mask_path)
        if mask_data is None:
            raise ValueError(f"Mask image not found in storage: {mask_path}")
        try:
            mask_array = np.frombuffer(mask_data.read(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1), np.uint8)
        finally:
            mask_data.close()
        mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot decode mask image: {mask_path}")
        mask = mask > 127
    else:
        # Create dummy mask for preview
        mask = np.zeros(img.shape[:2], bool)
        h, w = mask.shape
        mask[h//2:, w//4:3*w//4] = True
    
    gm_dict = {
        "H_i2w": np.array(ground_model.H_i2w, np.float64),
        "H_w2i": np.array(ground_model.H_w2i, np.float64),
        "R": np.array(ground_model.R_matrix, np.float64),
        "t": np.array(ground_model.t_vector, np.float64).reshape(3, 1),
        "K": np.array(calibration.k_matrix, np.float64),
        "cam_height": ground_model.cam_height,
        "pitch_down_deg": ground_model.pitch_down_deg,
    }
    
    bev_img, bev_mask, bev_overlay, bev_info = compute_bev(
        img, mask.astype(np.uint8), gm_dict, 
        ppm=ground_model.bev_ppm, margin_m=ground_model.bev_margin_m
    )
    
    # Save images
    import io
    
    def save_img(image: np.ndarray, suffix: str) -> str:
        buf = io.BytesIO(encode_image(image))
        key = generate_storage_key(
            str(camera.id), 
            output_type=f"bev_{suffix}", 
            extension="jpg"
        )
        storage.save(key, buf, "image/jpeg")
        return storage.get_url(key)
    
    return {
        "bev_image_url": save_img(bev_img, "image"),
        "bev_mask_url": save_img(bev_mask, "mask"),
        "bev_overlay_url": save_img(bev_overlay, "overlay"),
        "bev_info": bev_info,
    }


async def compute_depth_scale(
    db: AsyncSession,
    camera: Camera,
    calibration: Calibration,
    ground_model: GroundModel,
    test_image_key: str,
    depth_model: str,
    known_distances: List[dict],
    clip_min_m: float = 0.1,
    clip_max_m: float = 50.0,
) -> dict:
    """
    Compute depth scale from a test image with known distance measurements.
    
    Args:
        db: Database session
        camera: Camera object
        calibration: Active calibration
        ground_model: Active ground model
        test_image_key: Storage key of test image
        depth_model: Depth model name (e.g., 'yolo26n-depth.pt')
        known_distances: List of {x, y, distance_m} in undistorted image coordinates
        clip_min_m: Minimum depth clip
        clip_max_m: Maximum depth clip
    
    Returns:
        dict with scale_s, offset_b, residuals, mean_residual_m, max_residual_m
    """
    storage = get_storage()
    
    # Load test image from storage
    img_data = storage.get(test_image_key)
    if img_data is None:
        raise ValueError(f"Test image not found in storage: {test_image_key}")
    try:
        img_array = np.frombuffer(img_data.read(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1), np.uint8)
    finally:
        img_data.close()
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode test image: {test_image_key}")
    
    # Run depth estimation on test image
    from api.services.depth_service import run_depth
    depth_raw = await run_depth(img, depth_model)
    if depth_raw is None:
        raise ValueError("Depth estimation failed")
    
    # Undistort depth if calibration has original intrinsics
    if calibration.k_original is not None and calibration.resolution_w and calibration.resolution_h:
        K_orig = np.array(calibration.k_original, np.float64)
        dist = np.array(calibration.dist_coeffs, np.float64)
        K_target = np.array(calibration.k_undist or calibration.k_matrix, np.float64)
        # Undistort depth map
        depth_raw = cv2.undistort(depth_raw, K_orig, dist, None, K_target)
    
    # Extract depth values at known distance points
    raw_depths = []
    true_depths = []
    for pt in known_distances:
        x, y = int(pt["x"]), int(pt["y"])
        true_d = float(pt["distance_m"])
        if 0 <= y < depth_raw.shape[0] and 0 <= x < depth_raw.shape[1]:
            raw_d = float(depth_raw[y, x])
            if clip_min_m <= true_d <= clip_max_m and raw_d > 0:
                raw_depths.append(raw_d)
                true_depths.append(true_d)
    
    if len(raw_depths) < 2:
        raise ValueError(f"Need at least 2 valid known distance points within clip range, got {len(raw_depths)}")
    
    # Linear regression: true_depth = scale * raw_depth + offset
    raw_depths = np.array(raw_depths, np.float64)
    true_depths = np.array(true_depths, np.float64)
    
    A = np.vstack([raw_depths, np.ones_like(raw_depths)]).T
    scale_s, offset_b = np.linalg.lstsq(A, true_depths, rcond=None)[0]
    
    # Compute residuals
    pred_depths = scale_s * raw_depths + offset_b
    residuals = pred_depths - true_depths
    mean_residual = float(np.mean(np.abs(residuals)))
    max_residual = float(np.max(np.abs(residuals)))
    
    return {
        "scale_s": float(scale_s),
        "offset_b": float(offset_b),
        "residuals": residuals.tolist(),
        "mean_residual_m": mean_residual,
        "max_residual_m": max_residual,
        "clip_min_m": clip_min_m,
        "clip_max_m": clip_max_m,
        "depth_model": depth_model,
    }
