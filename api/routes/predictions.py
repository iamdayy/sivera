"""Prediction API routes."""
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
from sqlalchemy.orm import selectinload

from api.core.config import settings
from api.core.database import get_db
from api.core.security import verify_api_key, require_scope
from api.core.storage import get_storage, validate_image_content
from api.models.orm import (
    Camera, Calibration, GroundModel, DepthScale, 
    PredictionJob, PredictionOutput, DetectedInstance, VolumeEstimate
)
from api.schemas.prediction import (
    PredictionJobCreate, PredictionJobResponse, PredictionJobSubmitResponse,
    PredictionJobListResponse,
    PredictionOutputsResponse, PredictionOutputResponse,
    DetectedInstanceResponse, VolumeEstimateResponse
)
from api.services.prediction_orchestrator import submit_prediction_job
from api.workers.celery_app import celery_app


router = APIRouter(tags=["predictions"])


@router.post("", response_model=PredictionJobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_prediction(
    camera_id: UUID,
    source_type: str = Form(..., pattern="^(upload|rtsp_frame|latest_rtsp)$"),
    timestamp: Optional[datetime] = Form(None),
    seg_model: Optional[str] = Form(None),
    seg_conf: float = Form(0.25),
    seg_classes: Optional[str] = Form(None),  # JSON string
    volume_cell_m: float = Form(0.05),
    volume_min_pts_per_cell: int = Form(2),
    return_outputs: Optional[str] = Form(None),  # JSON string
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("predict:write")),
):
    """
    Submit a prediction job.
    
    For source_type="upload", a file must be provided.
    For source_type="rtsp_frame" or "latest_rtsp", the system will fetch from camera RTSP.
    """
    # Verify camera exists
    camera_result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = camera_result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Check active calibration, ground model, depth scale
    calib_result = await db.execute(
        select(Calibration).where(Calibration.camera_id == camera_id, Calibration.is_active == True)
    )
    if not calib_result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="No active calibration for camera")
    
    gr_result = await db.execute(
        select(GroundModel).join(GroundModel.ground_reference)
        .where(GroundModel.camera_id == camera_id, GroundReference.is_active == True)
    )
    if not gr_result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="No active ground model for camera")
    
    ds_result = await db.execute(
        select(DepthScale)
        .where(DepthScale.camera_id == camera_id)
        .order_by(DepthScale.created_at.desc())
    )
    if not ds_result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="No depth scale for camera")
    
    # Parse optional JSON fields
    import json
    seg_classes_list = None
    if seg_classes:
        try:
            seg_classes_list = json.loads(seg_classes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for seg_classes")
    
    return_outputs_list = None
    if return_outputs:
        try:
            return_outputs_list = json.loads(return_outputs)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for return_outputs")
    
    # Handle file upload
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
        
        # Save uploaded file
        storage = get_storage()
        import io
        content = await file.read()
        # Validate content (size + magic bytes)
        try:
            validate_image_content(content, file.filename or "upload",
                                   settings.MAX_UPLOAD_SIZE_MB)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        key = f"cameras/{camera_id}/uploads/{UUID(int=0).hex}_{file.filename}"
        buf = io.BytesIO(content)
        storage.save(key, buf, file.content_type or "image/jpeg")
        source_path = key
    
    # Create job
    job = await submit_prediction_job(
        db=db,
        camera_id=camera_id,
        source_type=source_type,
        source_path=source_path,
        source_ts=timestamp,
        seg_model=seg_model,
        seg_conf=seg_conf,
        seg_classes=seg_classes_list,
    )
    
    await db.commit()
    await db.refresh(job)
    
    # Queue prediction as Celery task (true asynchronous, matches 202 Accepted)
    try:
        celery_app.send_task(
            "api.workers.tasks.predict_task.run_prediction_task",
            args=[str(job.id)],
        )
    except Exception as e:
        # If queueing fails, mark job failed and return 503
        job.status = "failed"
        job.error_msg = "Prediction queue unavailable"
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
    result = await db.execute(
        select(VolumeEstimate)
        .where(VolumeEstimate.job_id == job_id)
        .order_by(VolumeEstimate.created_at)
    )
    volumes = result.scalars().all()
    
    return [VolumeEstimateResponse.model_validate(v) for v in volumes]