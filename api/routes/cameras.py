"""Camera API routes."""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
from sqlalchemy.orm import selectinload

from api.core.config import settings
from api.core.database import get_db
from api.core.security import verify_api_key, require_scope, check_camera_access
from api.routes.streams import get_mediamtx, stream_error
from api.services.mediamtx_service import MediaMTXError, MediaMTXService, validate_rtsp_source
from api.services.stream_service import reconcile_streams
from api.models.orm import ApiKey, Camera, Calibration
from api.schemas.camera import (
    CameraCreate, CameraUpdate, CameraResponse, CameraListResponse, CameraWithActiveCalib
)
from api.schemas.calibration import CalibrationResponse
from api.schemas.auto_prediction import AutoPredictConfig
from api.services.calibration_service import process_calibration_upload
from api.core.storage import get_storage, generate_storage_key


router = APIRouter(tags=["cameras"])


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreate,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:write")),
):
    """Register a new camera."""
    if api_key.camera_id is not None:
        raise HTTPException(status_code=403, detail="Camera-bound keys cannot register cameras")
    
    camera = Camera(
        name=camera_data.name,
        rtsp_url=camera_data.rtsp_url,
        resolution_w=camera_data.resolution_w,
        resolution_h=camera_data.resolution_h,
        fps=camera_data.fps,
        metadata_=camera_data.metadata,
    )
    
    # Set location if provided
    if camera_data.latitude is not None and camera_data.longitude is not None:
        from geoalchemy2 import WKTElement
        camera.location = WKTElement(
            f"POINT({camera_data.longitude} {camera_data.latitude})", srid=4326
        )
    
    # Set ROI polygon if provided
    if camera_data.roi_polygon:
        from geoalchemy2 import WKTElement
        coords = ", ".join(f"{lon} {lat}" for lon, lat in camera_data.roi_polygon)
        # Close the polygon
        if camera_data.roi_polygon[0] != camera_data.roi_polygon[-1]:
            coords += f", {camera_data.roi_polygon[0][0]} {camera_data.roi_polygon[0][1]}"
        camera.roi_polygon = WKTElement(f"POLYGON(({coords}))", srid=4326)
    
    db.add(camera)
    await db.flush()
    await db.refresh(camera)
    
    return CameraResponse.from_orm_with_geo(camera)


@router.get("", response_model=CameraListResponse)
async def list_cameras(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    active_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")),
):
    """List cameras with pagination."""
    total = 0
    items: List[CameraResponse] = []
    query = select(Camera)
    if api_key.camera_id is not None:
        query = query.where(Camera.id == api_key.camera_id)
    
    if active_only:
        # Join with calibrations to filter cameras with active calibration
        query = query.join(Calibration).where(Calibration.is_active == True).distinct()
    
    # Total count
    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar()
    if total == None or total == 0:
        return CameraListResponse(items=[], total=0, page=page, page_size=page_size, total_pages=0)
    
    # Paginated results
    query = query.order_by(Camera.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    cameras = result.scalars().all()
    items = [CameraResponse.from_orm_with_geo(c) for c in cameras]
    print("RTPS from items:", [item.rtsp_url for item in items])
    
    return CameraListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{camera_id}", response_model=CameraWithActiveCalib)
async def get_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")),
):
    """Get camera details with active calibration info."""
    check_camera_access(api_key, camera_id)
    result = await db.execute(
        select(Camera)
        .options(selectinload(Camera.calibrations))
        .where(Camera.id == camera_id)
    )
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Find active calibration
    active_calib = next((c for c in camera.calibrations if c.is_active), None)
    
    response = CameraWithActiveCalib.from_orm_with_geo(camera)
    if active_calib:
        response.active_calibration_id = active_calib.id
        response.active_calibration_version = active_calib.version
        response.active_calibration_rms = active_calib.rms_reproj
    
    return response


@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: UUID,
    camera_data: CameraUpdate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:write")),
    service: MediaMTXService = Depends(get_mediamtx),
):
    """Update camera metadata."""
    check_camera_access(api_key, camera_id)
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    update_data = camera_data.model_dump(exclude_unset=True)
    if "rtsp_url" in update_data and camera.stream_enabled:
        if update_data["rtsp_url"] is None:
            camera.stream_enabled = False
        else:
            try:
                validate_rtsp_source(update_data["rtsp_url"], service.config)
            except MediaMTXError as exc:
                raise stream_error(exc) from None
    
    # Handle geo fields separately
    if "latitude" in update_data or "longitude" in update_data:
        lat = update_data.pop("latitude", None)
        lon = update_data.pop("longitude", None)
        if lat is not None and lon is not None:
            from geoalchemy2 import WKTElement
            camera.location = WKTElement(f"POINT({lon} {lat})", srid=4326)
    
    if "roi_polygon" in update_data:
        roi = update_data.pop("roi_polygon", None)
        if roi:
            from geoalchemy2 import WKTElement
            coords = ", ".join(f"{lon} {lat}" for lon, lat in roi)
            if roi[0] != roi[-1]:
                coords += f", {roi[0][0]} {roi[0][1]}"
            camera.roi_polygon = WKTElement(f"POLYGON(({coords}))", srid=4326)
        else:
            camera.roi_polygon = None
    
    # Handle metadata separately (ORM attribute is metadata_, not metadata)
    if "metadata" in update_data:
        camera.metadata_ = update_data.pop("metadata")
    
    # Update remaining fields
    for key, value in update_data.items():
        if hasattr(camera, key):
            setattr(camera, key, value)
    
    await db.flush()
    await db.refresh(camera)
    await db.commit()
    background_tasks.add_task(reconcile_streams, service)
    return CameraResponse.from_orm_with_geo(camera)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:write")),
    service: MediaMTXService = Depends(get_mediamtx),
):
    """Delete camera (cascades to calibrations, jobs, etc.)."""
    check_camera_access(api_key, camera_id)
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    await db.delete(camera)
    await db.commit()
    background_tasks.add_task(reconcile_streams, service)
    # Cascade will handle related records


# Auto-predict config endpoints
@router.get("/{camera_id}/auto-predict")
async def get_auto_predict_config(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")),
):
    """Get auto-prediction configuration for a camera."""
    check_camera_access(api_key, camera_id)
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    config = camera.auto_predict_config or {}
    # Return with defaults filled in via the schema.
    validated = AutoPredictConfig.model_validate(config)
    return validated.model_dump(exclude_unset=False)


from api.schemas.auto_prediction import AutoPredictConfig, AutoPredictConfigUpdate


@router.patch("/{camera_id}/auto-predict")
async def update_auto_predict_config(
    camera_id: UUID,
    config_update: AutoPredictConfigUpdate,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:write")),
):
    """Update auto-prediction configuration for a camera.

    Only rtsp_frame source_type is supported for automatic inference. The
    runtime picks up the updated schedule on its next poll.
    """
    check_camera_access(api_key, camera_id)
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    current_config = camera.auto_predict_config or {}
    update_data = config_update.model_dump(exclude_unset=True)
    
    # Merge with current config, then validate via the schema.
    merged = {**current_config, **update_data}
    try:
        validated = AutoPredictConfig.model_validate(merged)
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail={"code": "prediction_invalid_parameters", "message": str(exc)}) from None
    
    new_config = validated.model_dump(exclude_unset=False)
    camera.auto_predict_config = new_config
    
    await db.flush()
    await db.commit()
    
    # Wake the runtime so it picks up the updated auto-predict schedule.
    try:
        from api.runtime.dependencies import notify_runtime
        notify_runtime(None)
    except Exception:
        pass  # Best effort
    
    return {
        "enabled": new_config.get("enabled", False),
        "interval_sec": new_config.get("interval_sec", settings.DEFAULT_AUTO_PREDICT_INTERVAL_SEC),
        "source_type": new_config.get("source_type", settings.DEFAULT_AUTO_PREDICT_SOURCE_TYPE),
        "seg_model": new_config.get("seg_model", settings.DEFAULT_AUTO_PREDICT_SEG_MODEL),
        "seg_conf": new_config.get("seg_conf", settings.DEFAULT_AUTO_PREDICT_SEG_CONF),
        "seg_classes": new_config.get("seg_classes"),
        "volume_cell_m": new_config.get("volume_cell_m", settings.DEFAULT_AUTO_PREDICT_VOLUME_CELL_M),
        "volume_min_pts_per_cell": new_config.get("volume_min_pts_per_cell", settings.DEFAULT_AUTO_PREDICT_VOLUME_MIN_PTS),
        "return_outputs": new_config.get("return_outputs", settings.DEFAULT_AUTO_PREDICT_RETURN_OUTPUTS),
        "max_frame_age_sec": new_config.get("max_frame_age_sec", settings.DEFAULT_AUTO_PREDICT_MAX_FRAME_AGE_SEC),
    }
