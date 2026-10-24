"""Ground Reference and Ground Model API routes."""
from typing import List, Optional
from uuid import UUID
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select, update
from sqlalchemy.orm import selectinload

from api.core.config import settings, STORAGE_ROOT
from api.core.database import get_db
from api.core.security import verify_api_key, require_scope
from api.models.orm import Camera, Calibration, GroundReference, GroundModel
from api.schemas.ground_ref import (
    GroundReferenceCreate, GroundReferenceResponse, GroundReferenceListResponse,
    GroundModelResponse, BEVPreviewResponse
)
from api.services.ground_model_service import (
    save_ground_reference, create_ground_model_from_ref, generate_bev_preview
)
from api.core.storage import get_storage


router = APIRouter(tags=["ground-references"])


@router.post("", response_model=GroundReferenceResponse, status_code=status.HTTP_201_CREATED)
async def create_ground_reference(
    camera_id: UUID,
    image_points: str = Form(...),  # JSON string: [[u,v],...]
    world_points: str = Form(...),  # JSON string: [[X,Y],...]
    world_units: str = Form("meter"),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:write")),
):
    """
    Submit ground reference points (image <-> world correspondences).
    
    This computes the ground model (homography + camera pose) automatically.
    """
    # Verify camera exists
    camera_result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = camera_result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Get active calibration
    calib_result = await db.execute(
        select(Calibration)
        .where(Calibration.camera_id == camera_id, Calibration.is_active == True)
    )
    calibration = calib_result.scalar_one_or_none()
    if not calibration:
        raise HTTPException(status_code=400, detail="No active calibration for this camera. Create one first.")
    
    # Parse JSON points
    import json
    try:
        img_pts = json.loads(image_points)
        world_pts = json.loads(world_points)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for points")
    
    if len(img_pts) != len(world_pts):
        raise HTTPException(status_code=400, detail="Number of image points must match world points")
    if len(img_pts) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 point correspondences")
    
    # Determine next version
    from sqlalchemy import func
    max_ver = await db.execute(
        select(func.max(GroundReference.version)).where(GroundReference.camera_id == camera_id)
    )
    next_version = (max_ver.scalar() or 0) + 1
    
    # Save ground reference
    ground_ref = await save_ground_reference(
        db=db,
        camera=camera,
        calibration=calibration,
        image_points=img_pts,
        world_points=world_pts,
        version=next_version,
        world_units=world_units,
        description=description,
    )
    
    # Create ground model
    ground_model = await create_ground_model_from_ref(
        db=db,
        camera=camera,
        calibration=calibration,
        ground_ref=ground_ref,
        ppm=settings.DEFAULT_BEV_PPM,
        margin_m=settings.DEFAULT_BEV_MARGIN_M,
    )
    
    await db.commit()
    await db.refresh(ground_ref)
    await db.refresh(ground_model)
    
    return GroundReferenceResponse.model_validate(ground_ref)


@router.get("", response_model=GroundReferenceListResponse)
async def list_ground_references(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """List all ground reference versions for a camera."""
    camera_result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = camera_result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    result = await db.execute(
        select(GroundReference)
        .where(GroundReference.camera_id == camera_id)
        .order_by(GroundReference.version.desc())
    )
    refs = result.scalars().all()
    
    return GroundReferenceListResponse(
        items=[GroundReferenceResponse.model_validate(r) for r in refs],
        total=len(refs),
    )


@router.get("/active", response_model=GroundReferenceResponse)
async def get_active_ground_reference(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """Get the currently active ground reference for a camera."""
    result = await db.execute(
        select(GroundReference)
        .where(GroundReference.camera_id == camera_id, GroundReference.is_active == True)
    )
    ground_ref = result.scalar_one_or_none()
    
    if not ground_ref:
        raise HTTPException(status_code=404, detail="No active ground reference for this camera")
    
    return GroundReferenceResponse.model_validate(ground_ref)


@router.get("/{ground_ref_id}", response_model=GroundReferenceResponse)
async def get_ground_reference(
    camera_id: UUID,
    ground_ref_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """Get specific ground reference version."""
    result = await db.execute(
        select(GroundReference)
        .where(GroundReference.id == ground_ref_id, GroundReference.camera_id == camera_id)
    )
    ground_ref = result.scalar_one_or_none()
    
    if not ground_ref:
        raise HTTPException(status_code=404, detail="Ground reference not found")
    
    return GroundReferenceResponse.model_validate(ground_ref)


@router.post("/{ground_ref_id}/activate", response_model=GroundReferenceResponse)
async def activate_ground_reference(
    camera_id: UUID,
    ground_ref_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:write")),
):
    """Set a ground reference version as active (deactivates others)."""
    result = await db.execute(
        select(GroundReference)
        .where(GroundReference.id == ground_ref_id, GroundReference.camera_id == camera_id)
    )
    ground_ref = result.scalar_one_or_none()
    
    if not ground_ref:
        raise HTTPException(status_code=404, detail="Ground reference not found for this camera")
    
    # Deactivate all
    await db.execute(
        update(GroundReference)
        .where(GroundReference.camera_id == camera_id)
        .values(is_active=False)
    )
    
    # Activate selected
    ground_ref.is_active = True
    await db.flush()
    await db.refresh(ground_ref)
    
    # Also need to recompute ground model with new active reference
    # Get active calibration
    calib_result = await db.execute(
        select(Calibration)
        .where(Calibration.camera_id == camera_id, Calibration.is_active == True)
    )
    calibration = calib_result.scalar_one_or_none()
    
    if calibration:
        # Remove old ground models for this ground reference to avoid accumulation
        await db.execute(
            delete(GroundModel).where(GroundModel.ground_ref_id == ground_ref.id)
        )
        # Create new ground model
        await create_ground_model_from_ref(
            db=db,
            camera=await db.get(Camera, camera_id),
            calibration=calibration,
            ground_ref=ground_ref,
            ppm=settings.DEFAULT_BEV_PPM,
            margin_m=settings.DEFAULT_BEV_MARGIN_M,
        )
    
    await db.commit()
    await db.refresh(ground_ref)
    
    return GroundReferenceResponse.model_validate(ground_ref)


# Ground Model routes
model_router = APIRouter(tags=["ground-models"])


@model_router.get("/active", response_model=GroundModelResponse)
async def get_active_ground_model(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """Get the ground model associated with active ground reference."""
    # Get active ground reference
    gr_result = await db.execute(
        select(GroundReference)
        .where(GroundReference.camera_id == camera_id, GroundReference.is_active == True)
    )
    ground_ref = gr_result.scalar_one_or_none()
    
    if not ground_ref:
        raise HTTPException(status_code=404, detail="No active ground reference")
    
    # Get ground model for this reference
    gm_result = await db.execute(
        select(GroundModel)
        .where(GroundModel.ground_ref_id == ground_ref.id)
    )
    ground_model = gm_result.scalar_one_or_none()
    
    if not ground_model:
        raise HTTPException(status_code=404, detail="Ground model not found for active reference")
    
    return GroundModelResponse.model_validate(ground_model)


@model_router.get("/{ground_model_id}", response_model=GroundModelResponse)
async def get_ground_model(
    camera_id: UUID,
    ground_model_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """Get specific ground model."""
    result = await db.execute(
        select(GroundModel)
        .where(GroundModel.id == ground_model_id, GroundModel.camera_id == camera_id)
    )
    ground_model = result.scalar_one_or_none()
    
    if not ground_model:
        raise HTTPException(status_code=404, detail="Ground model not found")
    
    return GroundModelResponse.model_validate(ground_model)


@model_router.get("/{ground_model_id}/bev-preview", response_model=BEVPreviewResponse)
async def get_bev_preview(
    camera_id: UUID,
    ground_model_id: UUID,
    test_image_path: Optional[str] = Query(None, description="Path to test image (optional)"),
    mask_path: Optional[str] = Query(None, description="Path to mask image (optional)"),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """Generate BEV preview for a ground model."""
    # Verify camera
    camera_result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = camera_result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Get ground model
    gm_result = await db.execute(
        select(GroundModel)
        .where(GroundModel.id == ground_model_id, GroundModel.camera_id == camera_id)
    )
    ground_model = gm_result.scalar_one_or_none()
    if not ground_model:
        raise HTTPException(status_code=404, detail="Ground model not found")
    
    # Get calibration
    calib_result = await db.execute(
        select(Calibration)
        .where(Calibration.id == ground_model.calibration_id)
    )
    calibration = calib_result.scalar_one_or_none()
    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found")

    # Validate optional file paths: only allow paths under STORAGE_ROOT
    def _validate_storage_path(p: Optional[str], label: str) -> Optional[str]:
        if not p:
            return None
        try:
            root = STORAGE_ROOT.resolve()
            candidate = Path(p).resolve()
            candidate.relative_to(root)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"{label} must be a path under the storage root",
            )
        if not candidate.exists():
            raise HTTPException(status_code=400, detail=f"{label} not found")
        return str(candidate)

    # Use default test image if not provided
    if not test_image_path:
        # Try to find a calibration image
        if calibration.calib_images:
            test_image_path = calibration.calib_images[0]
        else:
            raise HTTPException(status_code=400, detail="No test image available. Provide test_image_path.")
    else:
        test_image_path = _validate_storage_path(test_image_path, "test_image_path")
    if mask_path:
        mask_path = _validate_storage_path(mask_path, "mask_path")

    try:
        preview = await generate_bev_preview(
            db=db,
            camera=camera,
            calibration=calibration,
            ground_model=ground_model,
            test_image_path=test_image_path,
            mask_path=mask_path,
        )
        return BEVPreviewResponse(**preview)
    except Exception as e:
        raise HTTPException(status_code=500, detail="BEV preview generation failed. See server logs for details.")