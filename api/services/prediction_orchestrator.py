"""Prediction orchestrator: chains all services for the predict pipeline."""
import io
from datetime import datetime
from typing import List, Optional
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.core.storage import get_storage, generate_storage_key
from api.models.orm import (
    Camera, Calibration, GroundModel, GroundReference, DepthScale,
    PredictionJob, PredictionOutput
)
from api.services.image_service import encode_image
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
    1. Load the prerequisite versions pinned at submission
    2. Undistort image
    3. Segmentation (YOLO-seg) -> mask + instances
    4. Depth prediction -> scale -> lift to ground
    5. Volume estimation (grid + convex hull + per-instance)
    6. BEV projection
    7. Save all outputs and metadata
    """
    storage = get_storage()

    def add_output(output: PredictionOutput) -> None:
        if job.return_outputs is None or output.output_type in job.return_outputs:
            db.add(output)
    
    # Load related objects
    calibration = await db.get(Calibration, job.calibration_id)
    ground_model = await db.get(GroundModel, job.ground_model_id)
    depth_scale = await db.get(DepthScale, job.depth_scale_id)
    
    if not calibration or not ground_model or not depth_scale:
        raise ValueError("prediction_prerequisites_missing")
    if (calibration.camera_id != job.camera_id or ground_model.camera_id != job.camera_id
            or ground_model.calibration_id != calibration.id or depth_scale.camera_id != job.camera_id
            or depth_scale.ground_model_id != ground_model.id):
        raise ValueError("prediction_prerequisites_missing")
    
    camera = await db.get(Camera, job.camera_id)
    if camera is None:
        raise ValueError("prediction_prerequisites_missing")
    undist = prepare_prediction_image(image, calibration, camera, job.source_type)
    # Release prerequisite reads before model inference; completion is committed by the worker.
    await db.commit()
    
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
    add_output(mask_output)
    
    overlay_output = PredictionOutput(
        job_id=job.id, output_type="overlay",
        file_path=seg_results["overlay_key"], mime_type="image/jpeg",
        width=undist.shape[1], height=undist.shape[0],
    )
    add_output(overlay_output)
    
    instances_output = PredictionOutput(
        job_id=job.id, output_type="instances_json",
        file_path=seg_results["instances_json_key"], mime_type="application/json",
    )
    add_output(instances_output)
    
    # 3. DEPTH PIPELINE
    depth_results = await run_depth_pipeline(
        db, camera, calibration, ground_model, depth_scale,
        undist, waste_mask.astype(np.uint8), str(job.id)
    )
    
    # Save depth outputs
    for out_type, key in [
        ("depth_raw_vis", depth_results["depth_raw_key"]),
        ("depth_scaled_vis", depth_results["depth_scaled_key"]),
        ("depth_masked", depth_results["depth_masked_key"]),
        ("height_map", depth_results["height_map_key"]),
    ]:
        add_output(PredictionOutput(
            job_id=job.id, output_type=out_type, file_path=key,
            mime_type="image/jpeg", width=undist.shape[1], height=undist.shape[0],
        ))
    
    lifted_output = PredictionOutput(
        job_id=job.id, output_type="lifted_points_npz",
        file_path=depth_results["lifted_points_key"],
        mime_type="application/octet-stream",
    )
    add_output(lifted_output)
    
    # 4. VOLUME ESTIMATION
    # Load lifted points
    with storage.get(depth_results["lifted_points_key"]) as stream, np.load(stream, allow_pickle=False) as lifted_data:
        X, Y, H, uv = lifted_data["X"], lifted_data["Y"], lifted_data["height"], lifted_data["uv"]
    
    volume_results = await save_volume_results(
        db, job, X, Y, H, uv, instances, undist.shape[:2],
        cell_m=job.volume_cell_m, min_pts_per_cell=job.volume_min_pts_per_cell
    )
    
    # Save volume outputs
    add_output(PredictionOutput(
        job_id=job.id, output_type="volume_heatmap",
        file_path=volume_results["volume_heatmap_key"], mime_type="image/jpeg",
    ))
    add_output(PredictionOutput(
        job_id=job.id, output_type="volume_result_npz",
        file_path=volume_results["volume_npz_key"], mime_type="application/octet-stream",
    ))
    
    # 5. BEV PROJECTION
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
        buf = io.BytesIO(encode_image(img_data, ".jpg"))
        storage.save(key, buf, "image/jpeg")
        
        add_output(PredictionOutput(
            job_id=job.id, output_type=out_type, file_path=key,
            mime_type="image/jpeg", width=img_data.shape[1], height=img_data.shape[0],
            meta={"bev_info": bev_info} if out_type == "bev_image" else None,
        ))
    
    # 6. Build summary and update job
    summary = build_job_summary(waste_mask, instances, volume_results, depth_results)
    
    return summary


def prepare_prediction_image(image: np.ndarray, calibration: Calibration, camera: Camera, source_type: str) -> np.ndarray:
    if image is None:
        raise ValueError("prediction_invalid_image")
    size = (calibration.resolution_w, calibration.resolution_h)
    if calibration.k_original is not None and not all(size):
        raise ValueError("calibration_invalid_intrinsics")
    explicit = calibration.k_original is not None and all(size)
    if source_type != "upload" and not explicit:
        raise ValueError("calibration_intrinsics_missing")
    expected = size if explicit else (camera.resolution_w, camera.resolution_h)
    if image.shape[1::-1] != expected:
        raise ValueError("prediction_resolution_mismatch")
    registered = np.array(calibration.k_matrix, np.float64)
    if (registered.shape != (3, 3) or not np.isfinite(registered).all()
            or min(registered[0, 0], registered[1, 1]) <= 0
            or not np.allclose(registered[2], [0, 0, 1])):
        raise ValueError("calibration_invalid_intrinsics")
    if not explicit:
        # Legacy uploads retain the historical already-undistorted input contract.
        return image
    original = np.array(calibration.k_original, np.float64)
    target = np.array(calibration.k_undist or calibration.k_matrix, np.float64)
    distortion = np.array(calibration.dist_coeffs, np.float64).ravel()
    if (original.shape != (3, 3) or target.shape != (3, 3) or len(distortion) not in {4, 5, 8, 12, 14}
            or not np.isfinite(original).all() or not np.isfinite(target).all() or not np.isfinite(distortion).all()
            or min(original[0, 0], original[1, 1], target[0, 0], target[1, 1]) <= 0
            or not np.allclose(original[2], [0, 0, 1]) or not np.allclose(target, registered)):
        raise ValueError("calibration_invalid_intrinsics")
    return cv2.undistort(image, original, distortion, None, target)


async def submit_prediction_job(
    db: AsyncSession,
    camera_id: UUID,
    source_type: str,
    source_path: Optional[str] = None,
    source_ts: Optional[datetime] = None,
    seg_model: Optional[str] = None,
    seg_conf: float = 0.25,
    seg_classes: Optional[List[str]] = None,
    volume_cell_m: float = 0.05,
    volume_min_pts_per_cell: int = 2,
    return_outputs: Optional[List[str]] = None,
    max_frame_age_sec: int = 30,
    source_hash: str | None = None,
) -> PredictionJob:
    """Pin compatible prerequisite versions; the route dispatches after commit."""
    # Get active calibration, ground model, depth scale for camera
    calib_result = await db.execute(
        select(Calibration).where(Calibration.camera_id == camera_id, Calibration.is_active == True)
        .order_by(Calibration.version.desc()).limit(1)
    )
    calibration = calib_result.scalar_one_or_none()
    if not calibration:
        raise ValueError("prediction_requires_calibration")
    if source_type != "upload" and (calibration.k_original is None or not calibration.resolution_w or not calibration.resolution_h):
        raise ValueError("calibration_intrinsics_missing")
    
    gr_result = await db.execute(
        select(GroundModel).join(GroundModel.ground_reference)
        .where(GroundModel.camera_id == camera_id, GroundModel.calibration_id == calibration.id,
               GroundReference.camera_id == camera_id, GroundReference.calibration_id == calibration.id,
               GroundReference.is_active == True)
        .order_by(GroundReference.version.desc(), GroundModel.created_at.desc()).limit(1)
    )
    ground_model = gr_result.scalar_one_or_none()
    if not ground_model:
        raise ValueError("prediction_requires_ground_model")
    
    ds_result = await db.execute(
        select(DepthScale)
        .where(DepthScale.camera_id == camera_id, DepthScale.ground_model_id == ground_model.id, DepthScale.is_active == True)
        .order_by(DepthScale.created_at.desc()).limit(1)
    )
    depth_scale = ds_result.scalar_one_or_none()
    if not depth_scale:
        raise ValueError("prediction_requires_depth_scale")
    
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
        return_outputs=return_outputs,
        volume_cell_m=volume_cell_m, volume_min_pts_per_cell=volume_min_pts_per_cell,
        max_frame_age_sec=max_frame_age_sec, source_hash=source_hash,
    )
    db.add(job)
    await db.flush()
    
    return job
