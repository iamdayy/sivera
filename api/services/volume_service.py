"""Volume estimation service: wraps volume.py logic."""
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.storage import get_storage, generate_storage_key
from api.models.orm import PredictionJob, DetectedInstance, VolumeEstimate


def estimate_volume_grid(
    X: np.ndarray, Y: np.ndarray, height: np.ndarray,
    cell_m: float = 0.05, min_pts_per_cell: int = 2
) -> Tuple[float, float, np.ndarray, dict]:
    """Estimate volume using grid median method."""
    if len(height) == 0:
        return 0.0, 0.0, np.zeros((1, 1)), {}
    
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    
    nx = max(1, int(np.ceil((x_max - x_min) / cell_m)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell_m)))
    
    ix = np.clip(((X - x_min) / cell_m).astype(int), 0, nx - 1)
    iy = np.clip(((Y - y_min) / cell_m).astype(int), 0, ny - 1)
    
    grid_height = np.full((ny, nx), np.nan, dtype=np.float32)
    cell_area = cell_m ** 2
    
    for j in range(ny):
        for i in range(nx):
            mask = (ix == i) & (iy == j)
            pts = height[mask]
            if len(pts) >= min_pts_per_cell:
                grid_height[j, i] = float(np.median(pts))
    
    valid = np.isfinite(grid_height) & (grid_height > 0)
    volume_m3 = float(np.nansum(grid_height[valid]) * cell_area)
    footprint_m2 = float(valid.sum() * cell_area)
    
    grid_info = dict(
        x_min=x_min, y_min=y_min, cell_m=cell_m,
        nx=nx, ny=ny, shape=(ny, nx)
    )
    return volume_m3, footprint_m2, grid_height, grid_info


def estimate_volume_convexhull(X: np.ndarray, Y: np.ndarray, height: np.ndarray) -> Tuple[float, float]:
    """Estimate volume using convex hull method."""
    if len(height) < 3:
        return 0.0, 0.0
    
    try:
        pts2d = np.stack([X, Y], axis=1)
        hull = ConvexHull(pts2d)
        footprint_m2 = float(hull.volume)
    except Exception:
        footprint_m2 = float((X.max() - X.min()) * (Y.max() - Y.min()))
    
    med_h = float(np.median(height))
    volume_m3 = footprint_m2 * med_h
    return volume_m3, footprint_m2


def _poly_to_mask(poly: List[List[float]], shape: Tuple[int, int]) -> np.ndarray:
    """Convert polygon to binary mask."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(poly, np.float32)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def estimate_per_instance(
    instances: List[dict], lifted_npz: dict, img_shape: Tuple[int, int],
    cell_m: float = 0.05, min_pts_per_cell: int = 2
) -> List[dict]:
    """Estimate volume per instance."""
    X, Y, H, uv = lifted_npz["X"], lifted_npz["Y"], lifted_npz["height"], lifted_npz["uv"]
    H_img, W_img = img_shape[:2]
    
    results = []
    for ins in instances:
        poly = ins["polygon"]
        m = _poly_to_mask(poly, (H_img, W_img))
        
        u_arr, v_arr = uv[:, 0], uv[:, 1]
        in_inst = m[v_arr, u_arr]
        
        xi, yi, hi = X[in_inst], Y[in_inst], H[in_inst]
        n_pts = int(in_inst.sum())
        
        if n_pts < 3:
            vol, foot = 0.0, 0.0
        else:
            vol, foot, _, _ = estimate_volume_grid(xi, yi, hi, cell_m, min_pts_per_cell)
        
        results.append(dict(
            cls=ins["cls"],
            conf=ins["conf"],
            volume_m3=round(vol, 6),
            footprint_m2=round(foot, 6),
            n_pts=n_pts,
        ))
    return results


def render_volume_map(grid_height: np.ndarray, grid_info: dict, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Render grid height as heatmap."""
    gh = grid_height.copy()
    finite = np.isfinite(gh)
    if not finite.any():
        return np.zeros((*gh.shape, 3), np.uint8)
    
    lo, hi = 0.0, np.percentile(gh[finite], 98)
    if hi <= lo:
        hi = lo + 0.01
    
    norm = np.clip((gh - lo) / (hi - lo), 0.0, 1.0)
    norm[~finite] = 0.0
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, colormap)
    colored[~finite] = 0
    return colored


def add_scale_bar(img: np.ndarray, cell_m: float, ppm_display: Optional[float] = None) -> np.ndarray:
    """Add scale bar to image."""
    h, w = img.shape[:2]
    bar_m = 1.0
    if ppm_display is None:
        ppm_display = min(w, h) / 5.0
    bar_px = int(round(bar_m / cell_m))
    bar_px = min(bar_px, w // 3)
    
    x1, y1 = w - bar_px - 20, h - 30
    x2, y2 = x1 + bar_px, y1 + 10
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.putText(img, f"{bar_m:g} m", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


async def save_volume_results(
    db: AsyncSession,
    job: PredictionJob,
    X: np.ndarray, Y: np.ndarray, H: np.ndarray, uv: np.ndarray,
    instances: List[dict], img_shape: Tuple[int, int],
    cell_m: float, min_pts_per_cell: int,
) -> dict:
    """Save volume estimation results to database and storage."""
    storage = get_storage()
    
    # Grid method
    vol_grid, foot_grid, grid_h, ginfo = estimate_volume_grid(
        X, Y, H, cell_m, min_pts_per_cell
    )
    
    # Convex hull method
    vol_hull, foot_hull = estimate_volume_convexhull(X, Y, H)
    
    # Save volume estimates only when there is data.
    # Missing/failed estimates must remain missing, not represented as zero.
    if len(H) > 0:
        db.add(VolumeEstimate(
            job_id=job.id, method="grid",
            volume_m3=vol_grid, volume_liter=vol_grid * 1000,
            footprint_m2=foot_grid, cell_m=cell_m,
            height_mean_m=float(H.mean()),
            height_median_m=float(np.median(H)),
            n_cells=int(np.isfinite(grid_h).sum()), n_points=len(H),
        ))
        db.add(VolumeEstimate(
            job_id=job.id, method="convex_hull",
            volume_m3=vol_hull, volume_liter=vol_hull * 1000,
            footprint_m2=foot_hull, cell_m=None,
            height_mean_m=float(H.mean()),
            height_median_m=float(np.median(H)),
            n_cells=None, n_points=len(H),
        ))
    
    # Per-instance volumes
    inst_results = []
    if instances:
        inst_results = estimate_per_instance(
            instances, {"X": X, "Y": Y, "height": H, "uv": uv},
            img_shape, cell_m, min_pts_per_cell
        )
        for ir, ins in zip(inst_results, instances):
            db.add(DetectedInstance(
                job_id=job.id, cls_name=ir["cls"], confidence=ir["conf"],
                polygon_px=ins.get("polygon", []), polygon_world=[],
                volume_m3=ir["volume_m3"], footprint_m2=ir["footprint_m2"],
                height_mean_m=None, height_max_m=None, n_pixels=ir["n_pts"],
            ))
    
    # Save volume result NPZ
    vol_key = generate_storage_key(str(job.camera_id), str(job.id), "volume_result_npz", "npz")
    import io
    buf = io.BytesIO()
    np.savez(buf,
             vol_grid=vol_grid, foot_grid=foot_grid,
             vol_hull=vol_hull, foot_hull=foot_hull,
             grid_height=grid_h, cell_m=cell_m,
             x_min=ginfo.get("x_min", 0), y_min=ginfo.get("y_min", 0),
             nx=ginfo.get("nx", 0), ny=ginfo.get("ny", 0))
    buf.seek(0)
    storage.save(vol_key, buf, "application/octet-stream")
    
    # Save volume heatmap
    hmap = render_volume_map(grid_h, ginfo)
    hmap = add_scale_bar(hmap, cell_m)
    hmap_key = generate_storage_key(str(job.camera_id), str(job.id), "volume_heatmap", "jpg")
    buf = io.BytesIO()
    cv2.imwrite(buf, hmap, [cv2.IMWRITE_JPEG_QUALITY, 85])
    buf.seek(0)
    storage.save(hmap_key, buf, "image/jpeg")
    
    return {
        "volume_grid": {"volume_m3": vol_grid, "volume_liter": vol_grid * 1000, "footprint_m2": foot_grid, "cell_m": cell_m},
        "volume_convex_hull": {"volume_m3": vol_hull, "volume_liter": vol_hull * 1000, "footprint_m2": foot_hull},
        "instances": inst_results,
        "volume_npz_key": vol_key,
        "volume_heatmap_key": hmap_key,
    }


def build_job_summary(
    mask: np.ndarray, instances: List[dict],
    volume_results: dict, depth_results: dict
) -> dict:
    """Build summary for job result."""
    return {
        "waste_detected": bool(mask.any()),
        "total_instances": len(instances),
        "total_waste_pixels": int(mask.sum()),
        "waste_fraction_pct": round(100 * mask.mean(), 2),
        "volume": {
            "method_grid": volume_results["volume_grid"],
            "method_convex_hull": volume_results["volume_convex_hull"],
        },
        "instances": volume_results.get("instances", []),
        "depth": {
            "n_points_lifted": depth_results.get("n_points", 0),
            "height_stats": depth_results.get("height_stats", {}),
            "footprint": depth_results.get("footprint", {}),
        },
    }