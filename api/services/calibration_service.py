"""Calibration service: wraps existing calibrate.py logic for API use."""
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.storage import get_storage, generate_storage_key
from api.models.orm import Calibration, Camera

logger = logging.getLogger(__name__)


def calibrate_from_images(
    image_paths: List[Path],
    cols: int,
    rows: int,
    square_m: float,
    alpha: float = 0.0,
) -> dict:
    """
    Run camera calibration from chessboard images.
    Returns dict with calibration results.
    """
    pattern = (cols, rows)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    obj_pts, img_pts, used = [], [], []
    size = None
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]  # (w, h)
        elif gray.shape[::-1] != size:
            logger.debug("[skip] %s: size %s != %s", p, gray.shape[::-1], size)
            continue
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if not found:
            logger.debug("[skip] %s: chessboard not detected", p)
            continue
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(str(p))

    if len(used) < settings.MIN_CALIB_IMAGES:
        raise ValueError(f"Only {len(used)} valid images; minimum {settings.MIN_CALIB_IMAGES} required")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

    # Per-view errors
    per_view = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        d = proj.reshape(-1, 2) - i.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))))

    allc = np.concatenate([c.reshape(-1, 2) for c in img_pts])
    coverage = (
        (allc[:, 0].max() - allc[:, 0].min()) / size[0],
        (allc[:, 1].max() - allc[:, 1].min()) / size[1]
    )

    # Optimal new camera matrix (undistorted)
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)

    return {
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "size": list(size),
        "K_undist": K_new.tolist(),
        "rms": float(rms),
        "per_view": per_view,
        "used": used,
        "coverage": coverage,
        "alpha": alpha,
    }


def undistort_image(image: np.ndarray, K: np.ndarray, dist: np.ndarray, alpha: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort image and return undistorted image + K_undist."""
    h, w = image.shape[:2]
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (w, h), cv2.CV_16SC2)
    undist = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    return undist, K_new


def scale_K_for_resolution(K: np.ndarray, orig_size: Tuple[int, int], target_size: Tuple[int, int]) -> np.ndarray:
    """Scale camera matrix from original resolution to target resolution."""
    scale_x = target_size[0] / orig_size[0]
    scale_y = target_size[1] / orig_size[1]
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy
    return K_scaled


async def save_calibration_to_db(
    db: AsyncSession,
    camera: Camera,
    calib_result: dict,
    version: int,
    notes: Optional[str] = None,
    chessboard_cols: Optional[int] = None,
    chessboard_rows: Optional[int] = None,
    square_m: Optional[float] = None,
) -> Calibration:
    """Save calibration result to database."""
    # Deactivate previous active calibration for this camera
    from sqlalchemy import update
    await db.execute(
        update(Calibration)
        .where(Calibration.camera_id == camera.id)
        .values(is_active=False)
    )
    
    # Create new calibration record
    calibration = Calibration(
        camera_id=camera.id,
        version=version,
        k_matrix=calib_result["K_undist"],  # Store K_undist as primary K for undistorted images
        dist_coeffs=calib_result["dist"],
        rms_reproj=calib_result["rms"],
        coverage_w=calib_result["coverage"][0] * 100,
        coverage_h=calib_result["coverage"][1] * 100,
        chessboard_cols=chessboard_cols,
        chessboard_rows=chessboard_rows,
        square_m=square_m,
        calib_images=calib_result["used"],
        notes=notes,
        is_active=True,
    )
    db.add(calibration)
    await db.flush()
    return calibration


async def process_calibration_upload(
    db: AsyncSession,
    camera: Camera,
    image_files: List[Tuple[str, bytes]],  # (filename, content)
    cols: int = None,
    rows: int = None,
    square_m: float = None,
    alpha: float = 0.0,
    notes: Optional[str] = None,
) -> Calibration:
    """
    Process uploaded chessboard images and create calibration.
    
    Args:
        db: Database session
        camera: Camera object
        image_files: List of (filename, file_content) tuples
        cols: Chessboard inner corners width
        rows: Chessboard inner corners height
        square_m: Square size in meters
        alpha: Undistort alpha (0=crop, 1=keep all)
        notes: Optional notes
    
    Returns:
        Created Calibration object
    """
    cols = cols or settings.DEFAULT_CHESSBOARD_COLS
    rows = rows or settings.DEFAULT_CHESSBOARD_ROWS
    square_m = square_m or settings.DEFAULT_SQUARE_M
    
    # Save uploaded images to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        image_paths = []
        
        for filename, content in image_files:
            ext = Path(filename).suffix.lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            save_path = tmpdir_path / f"{uuid4().hex}{ext}"
            save_path.write_bytes(content)
            image_paths.append(save_path)
        
        if len(image_paths) < settings.MIN_CALIB_IMAGES:
            raise ValueError(f"Need at least {settings.MIN_CALIB_IMAGES} valid chessboard images")
        
        # Run calibration
        result = calibrate_from_images(image_paths, cols, rows, square_m, alpha)
        
        # Determine next version number
        from sqlalchemy import select, func
        max_ver = await db.execute(
            select(func.max(Calibration.version)).where(Calibration.camera_id == camera.id)
        )
        next_version = (max_ver.scalar() or 0) + 1
        
        # Save to database
        calibration = await save_calibration_to_db(
            db=db,
            camera=camera,
            calib_result=result,
            version=next_version,
            notes=notes,
            chessboard_cols=cols,
            chessboard_rows=rows,
            square_m=square_m,
        )
        
        # Also save undistort preview if we have a test image
        if image_paths:
            test_img = cv2.imread(str(image_paths[0]))
            undist, _ = undistort_image(
                test_img, 
                np.array(result["K"], np.float64), 
                np.array(result["dist"], np.float64),
                alpha
            )
            # Save preview to storage
            storage = get_storage()
            preview_key = generate_storage_key(str(camera.id), output_type="calib_preview", extension="jpg")
            import io
            # Use imencode to encode to memory buffer (imwrite doesn't support BytesIO in OpenCV 4.11+)
            success, buf_arr = cv2.imencode(".jpg", undist, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise RuntimeError("Failed to encode calibration preview image")
            buf = io.BytesIO(buf_arr.tobytes())
            storage.save(preview_key, buf, "image/jpeg")
        
        return calibration