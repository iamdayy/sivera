"""Celery tasks for async prediction processing."""
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from celery import shared_task
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.database import get_db_context
from api.core.storage import get_storage, generate_storage_key
from api.models.orm import (
    Camera, Calibration, GroundModel, DepthScale, 
    PredictionJob, PredictionOutput
)
from api.services.calibration_service import undistort_image
from api.services.segment_service import run_segmentation
from api.services.ground_model_service import compute_bev
from api.services.depth_service import run_depth_pipeline
from api.services.volume_service import save_volume_results, build_job_summary
from api.workers.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def run_prediction_task(self, job_id: str):
    """
    Celery task to run prediction pipeline asynchronously.
    """
    import asyncio
    
    async def _run():
        async with get_db_context() as db:
            job = await db.get(PredictionJob, job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            job.status = "processing"
            job.started_at = datetime.now(timezone.utc)
            await db.flush()
            
            try:
                # Load related objects
                calibration = await db.get(Calibration, job.calibration_id)
                ground_model = await db.get(GroundModel, job.ground_model_id)
                depth_scale = await db.get(DepthScale, job.depth_scale_id)
                camera = await db.get(Camera, job.camera_id)
                
                if not all([calibration, ground_model, depth_scale, camera]):
                    raise ValueError("Missing required calibration/ground_model/depth_scale/camera")
                
                # Load image
                storage = get_storage()
                if job.source_type == "upload" and job.source_path:
                    img_data = storage.get(job.source_path)
                    if not img_data:
                        raise ValueError("Failed to load uploaded image")
                    img_array = np.frombuffer(img_data.read(), np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    raise NotImplementedError("RTSP sources not yet implemented")
                
                # 1. UNDISTORT (placeholder - assumes already undistorted)
                undist = image
                
                # 2. SEGMENTATION
                seg_results = await run_segmentation(db, job, undist, None)
                instances = seg_results["instances"]
                waste_mask = seg_results["waste_mask"]
                
                # Save mask outputs
                for out_type, key, mime in [
                    ("mask", seg_results["mask_key"], "image/png"),
                    ("overlay", seg_results["overlay_key"], "image/jpeg"),
                    ("instances_json", seg_results["instances_json_key"], "application/json"),
                ]:
                    db.add(PredictionOutput(
                        job_id=job.id, output_type=out_type, file_path=key,
                        mime_type=mime, width=undist.shape[1], height=undist.shape[0],
                    ))
                
                # 3. DEPTH PIPELINE
                depth_results = await run_depth_pipeline(
                    db, camera, calibration, ground_model, depth_scale,
                    undist, waste_mask.astype(np.uint8), str(job.id)
                )
                
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
                
                db.add(PredictionOutput(
                    job_id=job.id, output_type="lifted_points_npz",
                    file_path=depth_results["lifted_points_key"],
                    mime_type="application/octet-stream",
                ))
                
                # 4. VOLUME ESTIMATION
                lifted_data = np.load(storage.get(depth_results["lifted_points_key"]))
                X, Y, H, uv = lifted_data["X"], lifted_data["Y"], lifted_data["height"], lifted_data["uv"]
                
                volume_results = await save_volume_results(
                    db, job, X, Y, H, uv, instances, undist.shape[:2],
                    cell_m=0.05, min_pts_per_cell=2
                )
                
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
                job.completed_at = datetime.now(timezone.utc)
                
                await db.commit()
                return summary
            
            except Exception as e:
                job.status = "failed"
                job.error_msg = str(e)
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()
                raise
    
    return asyncio.run(_run())


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_calibration_task(self, camera_id: str, image_files: list, cols: int, rows: int, square_m: float, alpha: float, notes: str):
    """
    Celery task to run camera calibration asynchronously.
    """
    import asyncio
    
    async def _run():
        async with get_db_context() as db:
            from api.services.calibration_service import process_calibration_upload
            from api.models.orm import Camera
            
            camera = await db.get(Camera, camera_id)
            if not camera:
                raise ValueError(f"Camera {camera_id} not found")
            
            # Process calibration
            calibration = await process_calibration_upload(
                db=db,
                camera=camera,
                image_files=image_files,
                cols=cols,
                rows=rows,
                square_m=square_m,
                alpha=alpha,
                notes=notes,
            )
            
            await db.commit()
            return str(calibration.id)
    
    return asyncio.run(_run())