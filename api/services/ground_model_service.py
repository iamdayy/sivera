"""Ground reference and ground model service: wraps ipm.py logic."""
from typing import List, Optional, Tuple
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update

from api.core.config import settings
from api.core.storage import get_storage, generate_storage_key
from api.models.orm import Camera, Calibration, GroundReference, GroundModel, PreviewArtifact, GroundReferenceSource
from api.services.image_service import encode_image
from api.services.depth_scale_service import compute_depth_scale, read_image, prepare_image
from api.services.ground_reference_source_service import read_source_image


def validate_ground_points(image_points: np.ndarray, world_points: np.ndarray,
                           size: tuple[int, int] | None = None) -> None:
    if (image_points.ndim != 2 or world_points.ndim != 2 or image_points.shape != world_points.shape
            or image_points.shape[1] != 2 or not 4 <= len(image_points) <= 100
            or not np.isfinite(image_points).all() or not np.isfinite(world_points).all()):
        raise ValueError("ground_reference_invalid_points")
    if size and ((image_points < 0).any() or (image_points >= np.asarray(size)).any()):
        raise ValueError("ground_reference_pixel_out_of_bounds")
    for points in (image_points, world_points):
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError("ground_reference_duplicate_points")
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if singular[1] <= singular[0] * 1e-6:
            raise ValueError("ground_reference_degenerate_points")


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
    
    validate_ground_points(image_points, world_points)
    K = np.asarray(K, np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("calibration_invalid_intrinsics")

    obj_points = np.hstack([world_points, np.zeros((len(world_points), 1))])
    try:
        H_w2i, _ = cv2.findHomography(world_points, image_points, 0)
        if H_w2i is None or not np.isfinite(H_w2i).all() or np.linalg.matrix_rank(H_w2i) < 3:
            raise ValueError("ground_reference_degenerate_points")
        H_i2w = np.linalg.inv(H_w2i)
        ok, rvec, tvec = cv2.solvePnP(obj_points, image_points, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            raise ValueError("ground_reference_invalid_geometry")
        rvec, tvec = cv2.solvePnPRefineLM(obj_points, image_points, K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, None)
    except (cv2.error, np.linalg.LinAlgError):
        raise ValueError("ground_reference_invalid_geometry") from None
    if (not all(np.isfinite(value).all() for value in (H_i2w, R, tvec, proj))
            or ((obj_points @ R.T + tvec.ravel())[:, 2] <= 0).any()):
        raise ValueError("ground_reference_invalid_geometry")
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
    bounds_world: Optional[np.ndarray] = None,
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
    
    if bounds_world is not None:
        xs, ys = np.asarray(bounds_world, np.float64).T
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError("bev_extent_invalid")
    xmin, xmax = xs.min() - margin_m, xs.max() + margin_m
    ymin, ymax = ys.min() - margin_m, ys.max() + margin_m
    
    wpx = int(round((xmax - xmin) * ppm))
    hpx = int(round((ymax - ymin) * ppm))
    
    if min(wpx, hpx) <= 0 or max(wpx, hpx) > 6000:
        raise ValueError("bev_extent_invalid")
    
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
    
    for index, wp in enumerate(gm.get("world_points", []), 1):
        pt = to_bev(wp)
        if 0 <= pt[0] < wpx and 0 <= pt[1] < hpx:
            cv2.circle(overlay, tuple(pt), 6, (0, 0, 255), -1)
            cv2.putText(overlay, str(index), (pt[0] + 9, pt[1] - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
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
    source_id: Optional[UUID] = None,
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
        source_id=source_id,
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
    test_image_path: Optional[str],
    mask_path: Optional[str] = None,
    source: Optional[GroundReferenceSource] = None,
    reference_points: Optional[List[List[float]]] = None,
) -> dict:
    """Generate BEV preview images and save to storage."""
    storage = get_storage()
    
    if source is not None:
        # Reuse the exact PNG shown during annotation, without another undistortion.
        img = read_source_image(source, storage)
    else:
        img = prepare_image(read_image(test_image_path, storage), calibration, camera)
    
    if mask_path:
        mask = cv2.cvtColor(read_image(mask_path, storage), cv2.COLOR_BGR2GRAY) > 127
        if mask.shape != img.shape[:2]:
            raise ValueError("bev_mask_resolution_mismatch")
    else:
        mask = np.zeros(img.shape[:2], bool)
    
    gm_dict = {
        "H_i2w": np.array(ground_model.H_i2w, np.float64),
        "H_w2i": np.array(ground_model.H_w2i, np.float64),
        "R": np.array(ground_model.R_matrix, np.float64),
        "t": np.array(ground_model.t_vector, np.float64).reshape(3, 1),
        "K": np.array(calibration.k_matrix, np.float64),
        "cam_height": ground_model.cam_height,
        "pitch_down_deg": ground_model.pitch_down_deg,
        "world_points": reference_points or [],
    }
    
    bev_img, bev_mask, bev_overlay, bev_info = compute_bev(
        img, mask.astype(np.uint8), gm_dict, 
        ppm=ground_model.bev_ppm, margin_m=ground_model.bev_margin_m,
        bounds_world=np.asarray(reference_points) if reference_points else None,
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
        db.add(PreviewArtifact(storage_key=key, ground_model_id=ground_model.id))
        return f"/files/{key}"
    
    bev_image_url = save_img(bev_img, "image")
    bev_mask_url = save_img(bev_mask, "mask")
    bev_overlay_url = save_img(bev_overlay, "overlay")
    
    return {
        "bev_image_url": bev_image_url,
        "bev_mask_url": bev_mask_url,
        "bev_overlay_url": bev_overlay_url,
        "bev_info": bev_info,
        "source_id": source.id if source else None,
        "has_mask": mask_path is not None,
    }
