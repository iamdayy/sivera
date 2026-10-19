"""Prediction orchestrator: chains all services for the predict pipeline."""
import io
import json
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from api.core.config import settings
from api.core.storage import get_storage, generate_storage_key
from api.models.orm import (
    Camera, Calibration, GroundModel, GroundReference, DepthScale,
    PredictionJob, PredictionOutput, DetectedInstance, VolumeEstimate
)
from api.services.calibration_service import undistort_image
from api.services.segment_service import run_segmentation
from api.services.ground_model_service import compute_bev
from api.services.depth_service import run_depth_pipeline
from api.services.volume_service import save_volume_results, build_job_summary


async def run_prediction_pipeline(
    db: AsyncSession,
    job: PredictionJob,
    image: np.ndarray,
) -> dict:
    """
    Run the complete prediction pipeline:
    1. Load active calibration, ground model, depth scale
    2. Undistort image
    3. Segmentation (YOLO-seg) -> mask + instances
    4. Depth prediction -> scale -> lift to ground
    5. Volume estimation (grid + convex hull + per-instance)
    6. BEV projection
    7. Save all outputs and metadata
    """
    storage = get_storage()
    
    # Load related objects
    calibration = await db.get(Calibration, job.calibration_id)
    ground_model = await db.get(GroundModel, job.ground_model_id)
    depth_scale = await db.get(DepthScale, job.depth_scale_id)
    
    if not calibration or not ground_model or not depth_scale:
        raise ValueError("Missing required calibration/ground_model/depth_scale")
    
    # 1. UNDISTORT image using calibration
    K_raw = np.array(calibration.k_matrix, np.float64)  # This is K_undist
    dist = np.array(calibration.dist_coeffs, np.float64)
    
    # Note: The stored k_matrix is already K_undist. For undistorting, we need original K.
    # Since we only have K_undist, we'll use it as the target for remap.
    # Actually, the calibration_service stores K_undist in k_matrix column.
    # We need the original K to undistort. This is a limitation.
    # For now, assume the input image is already undistorted or use K_undist as target.
    
    # If image size matches calibration size, it might already be undistorted
    calib_size = calibration.metadata_.get("original_size") if calibration.metadata_ else None
    # For simplicity, we'll assume the image needs undistorting with the stored K_undist
    # as if it were the target matrix. This is a known limitation.
    
    # Actually, the correct approach: we need original K and dist to undistort to K_undist.
    # Since we only saved K_undist, we can't properly undistort unless we also save original K.
    # For now, skip undistort and assume input is already undistorted at the right resolution.
    
    undist = image  # Placeholder - in production, implement proper undistort
    
    # 2. SEGMENTATION
    job.status = "processing"
    await db.flush()
    
    seg_results = await run_segmentation(db, job, undist, None)
    instances = seg_results["instances"]
    waste_mask = seg_results["waste_mask"]
    
    # Save mask output record
    mask_output = PredictionOutput(
        job_id=job.id, output_type="mask",
        file_path=seg_results["mask_key"], mime_type="image/png",
        width=undist.shape[1], height=undist.shape[0],
        meta={"waste_pixels": seg_results["total_waste_pixels"]}
    )
    db.add(mask_output)
    
    overlay_output = PredictionOutput(
        job_id=job.id, output_type="overlay",
        file_path=seg_results["overlay_key"], mime_type="image/jpeg",
        width=undist.shape[1], height=undist.shape[0],
    )
    db.add(overlay_output)
    
    instances_output = PredictionOutput(
        job_id=job.id, output_type="instances_json",
        file_path=seg_results["instances_json_key"], mime_type="application/json",
    )
    db.add(instances_output)
    
    # 3. DEPTH PIPELINE
    depth_results = await run_depth_pipeline(
        db, job.camera, calibration, ground_model, depth_scale,
        undist, waste_mask.astype(np.uint8), str(job.id)
    )
    
    # Save depth outputs
    for out_type, key in [
        ("depth_raw_vis", depth_results["depth_raw_key"]),
        ("depth_scaled_vis", depth_results["depth_scaled_key"]),
        ("depth_masked", depth_results["depth_masked_key"]),
        ("height_map", depth_results["height_map_key"]),
    ]:
        db.add(PredictionOutput(
            job_id=job.id, output_type=out_type, file_path=key,
            mime_type="image/jpeg", width=undist.shape[1], height=undist.shape[0],
        ))
    
    lifted_output = PredictionOutput(
        job_id=job.id, output_type="lifted_points_npz",
        file_path=depth_results["lifted_points_key"],
        mime_type="application/octet-stream",
    )
    db.add(lifted_output)
    
    # 4. VOLUME ESTIMATION
    # Load lifted points
    lifted_data = np.load(storage.get(depth_results["lifted_points_key"]))
    X, Y, H, uv = lifted_data["X"], lifted_data["Y"], lifted_data["height"], lifted_data["uv"]
    
    volume_results = await save_volume_results(
        db, job, X, Y, H, uv, instances, undist.shape[:2],
        cell_m=0.05, min_pts_per_cell=2
    )
    
    # Save volume outputs
    db.add(PredictionOutput(
        job_id=job.id, output_type="volume_heatmap",
        file_path=volume_results["volume_heatmap_key"], mime_type="image/jpeg",
    ))
    db.add(PredictionOutput(
        job_id=job.id, output_type="volume_result_npz",
        file_path=volume_results["volume_npz_key"], mime_type="application/octet-stream",
    ))
    
    # 5. BEV PROJECTION
    gm_dict = {
        "H_i2w": ground_model.H_i2w,
        "H_w2i": ground_model.H_w2i,
        "R": np.array(ground_model.R_matrix, np.float64),
        "t": np.array(ground_model.t_vector, np.float64).reshape(3, 1),
        "K": np.array(calibration.k_matrix, np.float64),
        "cam_height": ground_model.cam_height,
        "pitch_down_deg": ground_model.pitch_down_deg,
    }
    
    bev_img, bev_mask, bev_overlay, bev_info = compute_bev(
        undist, waste_mask.astype(np.uint8), gm_dict,
        ppm=ground_model.bev_ppm, margin_m=ground_model.bev_margin_m
    )
    
    # Save BEV outputs
    for out_type, img_data in [
        ("bev_image", bev_img),
        ("bev_mask_proyeksi", bev_mask),
        ("bev_overlay", bev_overlay),
    ]:
        key = generate_storage_key(str(job.camera_id), str(job.id), out_type, "jpg")
        buf = io.BytesIO()
        cv2.imwrite(buf, img_data, [cv2.IMWRITE_JPEG_QUALITY, 85])
        buf.seek(0)
        storage.save(key, buf, "image/jpeg")
        
        db.add(PredictionOutput(
            job_id=job.id, output_type=out_type, file_path=key,
            mime_type="image/jpeg", width=img_data.shape[1], height=img_data.shape[0],
            meta={"bev_info": bev_info} if out_type == "bev_image" else None,
        ))
    
    # 6. Build summary and update job
    summary = build_job_summary(waste_mask, instances, volume_results, depth_results)
    
    job.status = "done"
    job.result = summary
    
    await db.commit()
    
    return summary


async def submit_prediction_job(
    db: AsyncSession,
    camera_id: str,
    source_type: str,
    source_path: Optional[str] = None,
    source_ts: Optional[str] = None,
    seg_model: Optional[str] = None,
    seg_conf: float = 0.25,
    seg_classes: Optional[List[str]] = None,
    volume_cell_m: float = 0.05,
    volume_min_pts_per_cell: int = 2,
    return_outputs: Optional[List[str]] = None,
) -> PredictionJob:
    """
    Create and submit a prediction job.
    In production, this would queue a Celery task. For now, runs synchronously.
    """
    # Get active calibration, ground model, depth scale for camera
    calib_result = await db.execute(
        select(Calibration).where(Calibration.camera_id == camera_id, Calibration.is_active == True)
    )
    calibration = calib_result.scalar_one_or_none()
    if not calibration:
        raise ValueError("No active calibration for camera")
    
    gr_result = await db.execute(
        select(GroundModel).join(GroundModel.ground_reference)
        .where(GroundModel.camera_id == camera_id, GroundReference.is_active == True)
    )
    ground_model = gr_result.scalar_one_or_none()
    if not ground_model:
        raise ValueError("No active ground model for camera")
    
    ds_result = await db.execute(
        select(DepthScale)
        .where(DepthScale.camera_id == camera_id, DepthScale.ground_model_id == ground_model.id)
        .order_by(DepthScale.created_at.desc())
    )
    depth_scale = ds_result.scalar_one_or_none()
    if not depth_scale:
        raise ValueError("No depth scale for camera/ground_model")
    
    # Create job
    job = PredictionJob(
        camera_id=camera_id,
        calibration_id=calibration.id,
        ground_model_id=ground_model.id,
        depth_scale_id=depth_scale.id,
        status="pending",
        source_type=source_type,
        source_path=source_path,
        source_ts=source_ts,
        seg_model=seg_model,
        seg_conf=seg_conf,
        seg_classes=seg_classes,
    )
    db.add(job)
    await db.flush()
    
    return job