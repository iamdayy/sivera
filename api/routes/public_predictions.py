"""Public prediction routes: read-only access for dashboard visitors."""
import json
from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from api.core.database import get_db
from api.core.security import optional_scope
from api.core.storage import get_storage
from api.models.orm import ApiKey, Camera, PredictionJob, PredictionOutput, VolumeEstimate
from api.runtime import event_hub
from api.schemas.prediction import (
    PredictionJobResponse, PredictionJobListResponse
)


router = APIRouter(tags=["public-predictions"])


async def get_camera_or_404(db: AsyncSession, camera_id: UUID) -> Camera:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


async def _get_latest_prediction_from_db(db: AsyncSession, camera_id: UUID) -> dict | None:
    """Get latest completed prediction from database (no cache)."""
    # Find latest completed job
    result = await db.execute(
        select(PredictionJob)
        .options(
            selectinload(PredictionJob.outputs),
            selectinload(PredictionJob.volume_estimates),
        )
        .where(
            PredictionJob.camera_id == camera_id,
            PredictionJob.status == "done"
        )
        .order_by(PredictionJob.completed_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if not job:
        return None

    # Prepare response data
    volume_liters = None
    if job.volume_estimates:
        grid_vol = next((v for v in job.volume_estimates if v.method == "grid"), job.volume_estimates[0])
        volume_liters = grid_vol.volume_liter

    outputs = {out.output_type: out.file_path for out in job.outputs}
    
    return {
        "job_id": str(job.id),
        "camera_id": str(job.camera_id),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "volume_liters": volume_liters,
        "outputs": outputs,
        "seg_classes": job.seg_classes,
    }


@router.get("/cameras/{camera_id}/latest-prediction")
async def get_latest_prediction(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[ApiKey] = Depends(optional_scope("camera:read")),
):
    """Get the latest prediction for a camera from database.
    
    No Redis cache - reads directly from DB.
    No API key required for public access.
    """
    camera = await get_camera_or_404(db, camera_id)
    
    data = await _get_latest_prediction_from_db(db, camera_id)
    if not data:
        raise HTTPException(status_code=404, detail="No completed predictions")
    
    return data


@router.get("/cameras/{camera_id}/latest-prediction/outputs/{output_type}/image")
async def get_latest_prediction_output_image(
    camera_id: UUID,
    output_type: str,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[ApiKey] = Depends(optional_scope("camera:read")),
):
    """Get the latest prediction output image (streamed).

    Public endpoint with no-store cache headers.
    """
    camera = await get_camera_or_404(db, camera_id)
    
    # Find latest job with this output
    result = await db.execute(
        select(PredictionJob)
        .options(selectinload(PredictionJob.outputs))
        .where(
            PredictionJob.camera_id == camera_id,
            PredictionJob.status == "done"
        )
        .order_by(PredictionJob.completed_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="No completed predictions")
    
    output = next((o for o in job.outputs if o.output_type == output_type), None)
    if not output:
        raise HTTPException(status_code=404, detail=f"Output type '{output_type}' not found")
    storage_key = output.file_path
    
    storage = get_storage()
    stream = storage.get(storage_key)
    if stream is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # Determine mime type from output_type
    mime_types = {
        "mask": "image/png",
        "overlay": "image/jpeg",
        "bev_image": "image/jpeg",
        "bev_mask_proyeksi": "image/jpeg",
        "bev_overlay": "image/jpeg",
        "depth_raw_vis": "image/jpeg",
        "depth_scaled_vis": "image/jpeg",
        "depth_masked": "image/jpeg",
        "height_map": "image/jpeg",
        "volume_heatmap": "image/jpeg",
    }
    mime = mime_types.get(output_type, "application/octet-stream")
    
    # Cache headers: no-store for public access
    headers = {
        "Cache-Control": "no-store, private, must-revalidate",
        "Content-Type": mime,
    }
    
    def iter_stream():
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                yield chunk
        finally:
            stream.close()
    
    return StreamingResponse(iter_stream(), headers=headers, media_type=mime)


@router.get("/cameras/{camera_id}/predictions", response_model=PredictionJobListResponse)
async def list_public_predictions(
    camera_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    api_key: Optional[ApiKey] = Depends(optional_scope("camera:read")),
):
    """List prediction jobs for a camera (public, paginated)."""
    camera = await get_camera_or_404(db, camera_id)
    
    query = select(PredictionJob).where(PredictionJob.camera_id == camera_id)
    if status:
        query = query.where(PredictionJob.status == status)
    
    # Total count
    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar() or 0
    
    if total == 0:
        return PredictionJobListResponse(items=[], total=0, page=page, page_size=page_size, total_pages=0)
    
    query = query.order_by(PredictionJob.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return PredictionJobListResponse(
        items=[PredictionJobResponse.model_validate(j) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/cameras/{camera_id}/predictions/stream")
async def stream_prediction_updates(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: Optional[ApiKey] = Depends(optional_scope("camera:read")),
):
    """Server-Sent Events (SSE) stream for live prediction updates.
    
    Subscribes to the in-process event hub and forwards events.
    No authentication required for public access.
    """
    from sse_starlette.sse import EventSourceResponse

    camera = await get_camera_or_404(db, camera_id)

    async def event_generator():
        queue = event_hub.subscribe(camera_id)
        try:
            initial = await _get_latest_prediction_from_db(db, camera_id)
            if initial:
                yield {"event": "prediction", "data": json.dumps(initial)}
            while True:
                data = await queue.get()
                if data is None:
                    break
                yield {"event": "prediction", "data": data}
        finally:
            event_hub.unsubscribe(camera_id, queue)

    return EventSourceResponse(event_generator())