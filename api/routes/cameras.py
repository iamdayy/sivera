"""Camera API routes."""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
from sqlalchemy.orm import selectinload

from api.core.config import settings
from api.core.database import get_db
from api.core.security import verify_api_key, require_scope
from api.models.orm import Camera, Calibration
from api.schemas.camera import (
    CameraCreate, CameraUpdate, CameraResponse, CameraListResponse, CameraWithActiveCalib
)
from api.schemas.calibration import CalibrationResponse
from api.services.calibration_service import process_calibration_upload
from api.core.storage import get_storage, generate_storage_key


router = APIRouter(tags=["cameras"])


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreate,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("camera:write")),
):
    """Register a new camera."""
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
    api_key: str = Depends(require_scope("camera:read")),
):
    """List cameras with pagination."""
    query = select(Camera)
    
    if active_only:
        # Join with calibrations to filter cameras with active calibration
        query = query.join(Calibration).where(Calibration.is_active == True).distinct()
    
    # Total count
    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar()
    
    # Paginated results
    query = query.order_by(Camera.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    cameras = result.scalars().all()
    
    return CameraListResponse(
        items=[CameraResponse.from_orm_with_geo(c) for c in cameras],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/{camera_id}", response_model=CameraWithActiveCalib)
async def get_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("camera:read")),
):
    """Get camera details with active calibration info."""
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
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("camera:write")),
):
    """Update camera metadata."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    update_data = camera_data.model_dump(exclude_unset=True)
    
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
    
    return CameraResponse.from_orm_with_geo(camera)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("camera:write")),
):
    """Delete camera (cascades to calibrations, jobs, etc.)."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    await db.delete(camera)
    # Cascade will handle related records