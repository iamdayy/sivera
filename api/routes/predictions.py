"""Prediction API routes."""
import asyncio
import io
import json
from pathlib import Path
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
from sqlalchemy.orm import selectinload

from api.core.config import settings
from api.core.database import get_db
from api.core.security import check_camera_access, require_scope
from api.core.storage import get_storage, generate_storage_key
from api.models.orm import (
    ApiKey, Camera,
    PredictionJob, PredictionOutput, DetectedInstance, VolumeEstimate
)
from api.schemas.prediction import (
    PREDICTION_OUTPUT_TYPES, PredictionJobResponse, PredictionJobSubmitResponse,
    PredictionJobListResponse,
    PredictionOutputsResponse, PredictionOutputResponse,
    DetectedInstanceResponse, VolumeEstimateResponse
)
from api.services.prediction_orchestrator import submit_prediction_job
from api.workers.celery_app import celery_app
from api.services.image_service import decode_prediction_image
from api.services.prediction_access_service import require_job
from api.services.snapshot_service import latest_snapshot, require_rtsp_camera
from api.services.rtsp_capture_service import CaptureError
from api.services.calibration_capture_service import utcnow


router = APIRouter(tags=["predictions"])


@router.post("", response_model=PredictionJobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_prediction(
    camera_id: UUID,
    source_type: str = Form(..., pattern="^(upload|rtsp_frame|latest_rtsp)$",
                           description="upload: image file; rtsp_frame: fresh capture at worker execution; latest_rtsp: latest persisted snapshot at execution"),
    timestamp: Optional[datetime] = Form(None),
    seg_model: Optional[str] = Form(None),
    seg_conf: float = Form(0.25, ge=0, le=1, allow_inf_nan=False),
    seg_classes: Optional[str] = Form(None),  # JSON string
    volume_cell_m: float = Form(0.05, gt=0, le=1, allow_inf_nan=False),
    volume_min_pts_per_cell: int = Form(2, ge=1, le=10000),
    max_frame_age_sec: int = Form(30, ge=1, le=300, description="Maximum RTSP snapshot age in seconds, checked again by the worker"),
    return_outputs: Optional[str] = Form(None, description="JSON array of output types: omitted registers all, [] registers none. Internal artifacts may still be generated"),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("predict:write")),
):
    """Queue upload, worker-time RTSP capture, or a freshness-bounded persisted snapshot."""
    check_camera_access(api_key, camera_id)
    camera_result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = camera_result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    def string_list(value: str | None, field: str) -> list[str] | None:
        if value is None:
            return None
        try:
            result = json.loads(value)
            if not isinstance(result, list) or len(result) > 100 or any(not isinstance(item, str) or len(item) > 100 for item in result):
                raise ValueError()
            return result
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail={"code": "prediction_invalid_parameters", "field": field}) from None
    seg_classes_list = string_list(seg_classes, "seg_classes")
    return_outputs_list = string_list(return_outputs, "return_outputs")
    if return_outputs_list is not None and not set(return_outputs_list).issubset(PREDICTION_OUTPUT_TYPES):
        raise HTTPException(status_code=422, detail={"code": "prediction_invalid_outputs"})
    if seg_model and (Path(seg_model).name != seg_model or "\\" in seg_model or len(seg_model) > 100):
        raise HTTPException(status_code=422, detail={"code": "prediction_invalid_model"})
    identity = None
    if source_type != "upload":
        if file is not None or timestamp is not None:
            raise HTTPException(status_code=422, detail={"code": "rtsp_source_disallows_file_or_timestamp"})
        try:
            identity = require_rtsp_camera(camera)
            if source_type == "latest_rtsp":
                await latest_snapshot(db, camera_id, identity, max_frame_age_sec)
        except CaptureError as exc:
            raise HTTPException(status_code=409, detail={"code": exc.code}) from None
    source_path = None
    if source_type == "upload":
        if not file:
            raise HTTPException(status_code=400, detail="File required for upload source type")
        
        # Pre-read size check (from Content-Length header)
        if file.size is not None and file.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )
        
        storage = get_storage()
        content = await file.read(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
        try:
            await asyncio.to_thread(decode_prediction_image, content)
        except ValueError as exc:
            raise HTTPException(status_code=413 if str(exc) == "prediction_image_too_large" else 400, detail={"code": str(exc)}) from None
        source_path = generate_storage_key(str(camera_id), output_type="uploads", extension="image")
    
    # Create job
    try:
        job = await submit_prediction_job(
            db=db, camera_id=camera_id, source_type=source_type, source_path=source_path,
            source_ts=timestamp, seg_model=seg_model, seg_conf=seg_conf, seg_classes=seg_classes_list,
            volume_cell_m=volume_cell_m, volume_min_pts_per_cell=volume_min_pts_per_cell,
            return_outputs=return_outputs_list, max_frame_age_sec=max_frame_age_sec, source_hash=identity,
        )
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail={"code": str(exc)}) from None
    try:
        if source_path:
            await asyncio.to_thread(storage.save, source_path, io.BytesIO(content), file.content_type or "application/octet-stream")
        await db.commit()
    except Exception:
        await db.rollback()
        if source_path:
            try:
                await asyncio.to_thread(storage.delete, source_path)
            except Exception:
                pass
        raise HTTPException(status_code=503, detail={"code": "prediction_submission_failed"}) from None
    
    try:
        await asyncio.to_thread(celery_app.send_task,
            "api.workers.tasks.predict_task.run_prediction_task",
            args=[str(job.id)],
            queue="prediction", retry=False,
        )
    except Exception:
        # If queueing fails, mark job failed and return 503
        await db.execute(update(PredictionJob).where(PredictionJob.id == job.id, PredictionJob.status == "pending").values(
            status="failed", error_msg="prediction_queue_unavailable", completed_at=utcnow(),
        ))
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction queue unavailable. Please try again later.",
        )

    return PredictionJobSubmitResponse(
        job_id=job.id,
        status="pending",
        message="Prediction accepted for asynchronous processing",
        estimated_duration_sec=30,
    )


@router.get("/{job_id}", response_model=PredictionJobResponse)
async def get_prediction_job(
    camera_id: UUID,
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("predict:read")),
):
    """Get prediction job status and result."""
    check_camera_access(api_key, camera_id)
    result = await db.execute(
        select(PredictionJob)
        .options(
            selectinload(PredictionJob.outputs),
            selectinload(PredictionJob.instances),
            selectinload(PredictionJob.volume_estimates),
        )
        .where(PredictionJob.id == job_id, PredictionJob.camera_id == camera_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Prediction job not found")
    
    return PredictionJobResponse.model_validate(job)


@router.get("", response_model=PredictionJobListResponse)
async def list_prediction_jobs(
    camera_id: UUID,
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("predict:read")),
):
    """List prediction jobs for a camera."""
    check_camera_access(api_key, camera_id)
    query = select(PredictionJob).where(PredictionJob.camera_id == camera_id)
    
    if status:
        query = query.where(PredictionJob.status == status)
    
    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar()
    
    query = query.order_by(PredictionJob.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    total_pages = (total + page_size - 1) // page_size
    return PredictionJobListResponse(
        items=[PredictionJobResponse.model_validate(j) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{job_id}/outputs", response_model=PredictionOutputsResponse)
async def get_prediction_outputs(
    camera_id: UUID,
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("predict:read")),
):
    """List all output files for a prediction job."""
    await require_job(db, camera_id, job_id, api_key)
    result = await db.execute(
        select(PredictionOutput)
        .where(PredictionOutput.job_id == job_id)
        .order_by(PredictionOutput.created_at)
    )
    outputs = result.scalars().all()
    
    return PredictionOutputsResponse(
        outputs=[PredictionOutputResponse.model_validate(o) for o in outputs]
    )


@router.get("/{job_id}/outputs/{output_type}")
async def download_prediction_output(
    camera_id: UUID,
    job_id: UUID,
    output_type: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("predict:read")),
):
    """Download a specific output file."""
    await require_job(db, camera_id, job_id, api_key)
    result = await db.execute(
        select(PredictionOutput)
        .where(PredictionOutput.job_id == job_id, PredictionOutput.output_type == output_type)
    )
    output = result.scalar_one_or_none()
    
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")
    
    storage = get_storage()
    file_stream = storage.get(output.file_path)
    
    if not file_stream:
        raise HTTPException(status_code=404, detail="File not found in storage")
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        file_stream,
        media_type=output.mime_type or "application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={output_type}.{output.mime_type.split('/')[-1] if output.mime_type else 'bin'}"}
    )


@router.get("/{job_id}/instances", response_model=List[DetectedInstanceResponse])
async def get_prediction_instances(
    camera_id: UUID,
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("predict:read")),
):
    """Get detected instances for a prediction job."""
    await require_job(db, camera_id, job_id, api_key)
    result = await db.execute(
        select(DetectedInstance)
        .where(DetectedInstance.job_id == job_id)
        .order_by(DetectedInstance.created_at)
    )
    instances = result.scalars().all()
    
    return [DetectedInstanceResponse.model_validate(i) for i in instances]


@router.get("/{job_id}/volumes", response_model=List[VolumeEstimateResponse])
async def get_prediction_volumes(
    camera_id: UUID,
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("predict:read")),
):
    """Get volume estimates for a prediction job."""
    await require_job(db, camera_id, job_id, api_key)
    result = await db.execute(
        select(VolumeEstimate)
        .where(VolumeEstimate.job_id == job_id)
        .order_by(VolumeEstimate.created_at)
    )
    volumes = result.scalars().all()
    
    return [VolumeEstimateResponse.model_validate(v) for v in volumes]
