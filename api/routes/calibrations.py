"""Calibration API routes."""
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
from api.schemas.calibration import (
    CalibrationCreate, CalibrationResponse, CalibrationListResponse, 
    CalibrationActivateResponse, UndistortPreviewResponse
)
from api.services.calibration_service import process_calibration_upload, undistort_image, scale_K_for_resolution
from api.core.storage import get_storage, validate_image_content


router = APIRouter(tags=["calibrations"])


@router.post("", response_model=CalibrationResponse, status_code=status.HTTP_201_CREATED)
async def create_calibration(
    camera_id: UUID,
    chessboard_cols: int = Form(settings.DEFAULT_CHESSBOARD_COLS),
    chessboard_rows: int = Form(settings.DEFAULT_CHESSBOARD_ROWS),
    square_m: float = Form(settings.DEFAULT_SQUARE_M),
    alpha: float = Form(0.0),
    notes: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("calibration:write")),
):
    """
    Upload chessboard images and run calibration.
    
    Requires at least 10 valid chessboard images (recommended 15-30).
    Images should be JPG/PNG/BMP format.
    """
    # Verify camera exists
    camera_result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = camera_result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Read uploaded files
    image_files = []
    for f in files:
        # Pre-read size check (from Content-Length header)
        if f.size is not None and f.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{f.filename}' too large. Maximum {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )
        content = await f.read()
        # Validate content (size + magic bytes)
        try:
            validate_image_content(content, f.filename or "image",
                                   settings.MAX_UPLOAD_SIZE_MB)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if content:
            image_files.append((f.filename, content))
    
    if len(image_files) < settings.MIN_CALIB_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {settings.MIN_CALIB_IMAGES} valid images, got {len(image_files)}"
        )
    
    try:
        calibration = await process_calibration_upload(
            db=db,
            camera=camera,
            image_files=image_files,
            cols=chessboard_cols,
            rows=chessboard_rows,
            square_m=square_m,
            alpha=alpha,
            notes=notes,
        )
        await db.commit()
        await db.refresh(calibration)
        return CalibrationResponse.model_validate(calibration)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Calibration processing error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Calibration failed. See server logs for details.",
        )


@router.get("", response_model=CalibrationListResponse)
async def list_calibrations(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("calibration:read")),
):
    """List all calibration versions for a camera."""
    # Verify camera exists
    camera_result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = camera_result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    result = await db.execute(
        select(Calibration)
        .where(Calibration.camera_id == camera_id)
        .order_by(Calibration.version.desc())
    )
    calibrations = result.scalars().all()
    
    return CalibrationListResponse(
        items=[CalibrationResponse.model_validate(c) for c in calibrations],
        total=len(calibrations),
    )


@router.get("/active", response_model=CalibrationResponse)
async def get_active_calibration(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("calibration:read")),
):
    """Get the currently active calibration for a camera."""
    result = await db.execute(
        select(Calibration)
        .where(Calibration.camera_id == camera_id, Calibration.is_active == True)
    )
    calibration = result.scalar_one_or_none()
    
    if not calibration:
        raise HTTPException(status_code=404, detail="No active calibration for this camera")
    
    return CalibrationResponse.model_validate(calibration)


@router.get("/{calibration_id}", response_model=CalibrationResponse)
async def get_calibration(
    camera_id: UUID,
    calibration_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("calibration:read")),
):
    """Get specific calibration version."""
    result = await db.execute(
        select(Calibration)
        .where(Calibration.id == calibration_id, Calibration.camera_id == camera_id)
    )
    calibration = result.scalar_one_or_none()
    
    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    return CalibrationResponse.model_validate(calibration)


@router.post("/{calibration_id}/activate", response_model=CalibrationActivateResponse)
async def activate_calibration(
    camera_id: UUID,
    calibration_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("calibration:write")),
):
    """Set a calibration version as active (deactivates others)."""
    # Verify calibration belongs to camera
    result = await db.execute(
        select(Calibration)
        .where(Calibration.id == calibration_id, Calibration.camera_id == camera_id)
    )
    calibration = result.scalar_one_or_none()
    
    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found for this camera")
    
    # Deactivate all calibrations for this camera
    await db.execute(
        update(Calibration)
        .where(Calibration.camera_id == camera_id)
        .values(is_active=False)
    )
    
    # Activate selected
    calibration.is_active = True
    await db.flush()
    await db.refresh(calibration)
    
    return CalibrationActivateResponse(
        id=calibration.id,
        camera_id=calibration.camera_id,
        version=calibration.version,
        is_active=True,
        message=f"Calibration v{calibration.version} activated",
    )


@router.get("/{calibration_id}/undistort-preview", response_model=UndistortPreviewResponse)
async def get_undistort_preview(
    camera_id: UUID,
    calibration_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("calibration:read")),
):
    """Get undistort preview for a calibration (uses first calibration image)."""
    result = await db.execute(
        select(Calibration)
        .where(Calibration.id == calibration_id, Calibration.camera_id == camera_id)
    )
    calibration = result.scalar_one_or_none()
    
    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    if not calibration.calib_images:
        raise HTTPException(status_code=400, detail="No calibration images available")
    
    # Load first calibration image from permanent storage
    import cv2
    import numpy as np
    import io
    
    storage = get_storage()
    first_img_key = calibration.calib_images[0]
    img_data = storage.get(first_img_key)
    if img_data is None:
        raise HTTPException(status_code=400, detail="Calibration image not found in storage")
    
    img_array = np.frombuffer(img_data.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode calibration image")
    
    K = np.array(calibration.k_matrix, np.float64)
    dist = np.array(calibration.dist_coeffs, np.float64)
    
    storage = get_storage()
    preview_key = f"cameras/{camera_id}/calibrations/{calibration_id}/preview.jpg"
    
    if storage.exists(preview_key):
        return UndistortPreviewResponse(
            preview_url=storage.get_url(preview_key),
            original_size=[img.shape[1], img.shape[0]],
            undistorted_size=[img.shape[1], img.shape[0]],
            k_undist=calibration.k_matrix,
        )
    
    # Generate preview if not cached
    undist = cv2.undistort(img, K, dist, None, K)
    
    import io
    success, buf_arr = cv2.imencode(".jpg", undist, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode preview image")
    
    buf = io.BytesIO(buf_arr.tobytes())
    storage.save(preview_key, buf, "image/jpeg")
    
    return UndistortPreviewResponse(
        preview_url=storage.get_url(preview_key),
        original_size=[img.shape[1], img.shape[0]],
        undistorted_size=[img.shape[1], img.shape[0]],
        k_undist=calibration.k_matrix,
    )


@router.delete("/{calibration_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_calibration(
    camera_id: UUID,
    calibration_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("calibration:write")),
):
    """Delete a calibration version (cannot delete active)."""
    result = await db.execute(
        select(Calibration)
        .where(Calibration.id == calibration_id, Calibration.camera_id == camera_id)
    )
    calibration = result.scalar_one_or_none()
    
    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    if calibration.is_active:
        raise HTTPException(status_code=400, detail="Cannot delete active calibration. Activate another first.")
    
    await db.delete(calibration)
    await db.commit()