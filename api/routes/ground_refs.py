"""Ground Reference and Ground Model API routes."""
from typing import List, Optional
from uuid import UUID, uuid4
import io
import asyncio
import json
import logging
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, update
from pydantic import ValidationError

from api.core.config import settings
from api.core.database import get_db
from api.core.security import verify_api_key, require_scope, check_camera_access
from api.models.orm import ApiKey, Camera, Calibration, GroundReference, GroundModel, DepthScale, PreviewArtifact
from api.schemas.ground_ref import (
    GroundReferenceCreate, GroundReferenceResponse, GroundReferenceListResponse,
    GroundModelResponse, BEVPreviewResponse,
    DepthScaleCreate, DepthScaleResponse, DepthScaleListResponse, DepthScaleImageResponse,
    DepthScalePreviewCreate, DepthScalePreviewResponse
)
from api.services.ground_model_service import (
    save_ground_reference, create_ground_model_from_ref, generate_bev_preview, compute_depth_scale, validate_ground_points
)
from api.services.ground_reference_source_service import (
    GroundReferenceError, active_calibration, lock_camera, read_source_bytes, require_source, require_source_context,
)
from api.routes.ground_reference_sources import ground_reference_error
from api.core.storage import get_storage
from api.services.depth_scale_service import test_image_key, is_test_image_key, prepare_image, validate_saved_scale
from api.services.depth_model_service import validate_scale_model
from api.services.image_service import decode_prediction_image, encode_image
from api.services.vision_process_service import VisionError
from api.services.prediction_access_service import authorize_artifact


logger = logging.getLogger(__name__)


async def ground_camera_access(camera_id: UUID, api_key: ApiKey = Depends(verify_api_key)) -> None:
    check_camera_access(api_key, camera_id)


router = APIRouter(tags=["ground-references"], dependencies=[Depends(ground_camera_access)])


@router.post("", response_model=GroundReferenceResponse, status_code=status.HTTP_201_CREATED, responses={
    400: {"description": "Invalid point pairs, units, geometry or out-of-bounds pixels"},
    404: {"description": "Camera or source missing"},
    409: {"description": "Active calibration, RTSP source or dimensions changed"},
    503: {"description": "Persistence unavailable; check reference history before retrying"},
})
async def create_ground_reference(
    camera_id: UUID,
    image_points: str = Form(..., max_length=32768, description="JSON array of 4..100 undistorted [u,v] pixel pairs"),
    world_points: str = Form(..., max_length=32768, description="JSON array of corresponding [X,Y] pairs in meters"),
    world_units: str = Form("meter", pattern="^(meter|metre)$"),
    description: Optional[str] = Form(None, max_length=2000),
    source_id: Optional[UUID] = Form(None, description="Prepared snapshot source; omitted retains legacy undistorted-point input"),
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("ground_ref:write")),
):
    """Create and activate a reference and its ground model atomically (201).

    Source-bound points are validated against the exact frozen image and active
    calibration under a camera lock. This operation is not a queued job or an
    idempotent write; check history after an ambiguous transport failure.
    """
    try:
        body = GroundReferenceCreate(image_points=json.loads(image_points), world_points=json.loads(world_points),
            source_id=source_id, world_units=world_units, description=description)
    except (json.JSONDecodeError, ValidationError):
        raise HTTPException(status_code=400, detail={"code": "ground_reference_invalid_points"}) from None

    try:
        camera = await lock_camera(db, camera_id)
        source = await require_source(db, camera_id, source_id) if source_id else None
        calibration = await active_calibration(db, camera_id, source.calibration_id if source else None)
        if source is not None:
            require_source_context(camera, calibration, source)
            await asyncio.to_thread(read_source_bytes, source, get_storage())
        size = (source.width, source.height) if source else (
            calibration.resolution_w or camera.resolution_w, calibration.resolution_h or camera.resolution_h)
        img_pts, world_pts = np.asarray(body.image_points, np.float64), np.asarray(body.world_points, np.float64)
        validate_ground_points(img_pts, world_pts, size)
        version = (await db.scalar(select(func.max(GroundReference.version)).where(GroundReference.camera_id == camera_id)) or 0) + 1
        ground_ref = await save_ground_reference(db, camera, calibration, img_pts.tolist(), world_pts.tolist(),
            version, world_units=body.world_units, description=body.description, source_id=source_id)
        await create_ground_model_from_ref(db, camera, calibration, ground_ref,
            ppm=settings.DEFAULT_BEV_PPM, margin_m=settings.DEFAULT_BEV_MARGIN_M)
        await db.commit()
        await db.refresh(ground_ref)
        return GroundReferenceResponse.model_validate(ground_ref)
    except (GroundReferenceError, ValueError) as exc:
        await db.rollback()
        raise ground_reference_error(exc) from None
    except Exception:
        await db.rollback()
        logger.error("Ground reference persistence failed for camera %s", camera_id)
        raise HTTPException(status_code=503, detail={"code": "ground_reference_save_unavailable"}) from None


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


@router.get("/{ground_ref_id}/ground-model", response_model=GroundModelResponse)
async def get_reference_ground_model(camera_id: UUID, ground_ref_id: UUID,
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("ground_ref:read")),
) -> GroundModelResponse:
    """Read the saved model for this reference, including inactive history."""
    model = (await db.execute(select(GroundModel).join(GroundReference, GroundModel.ground_ref_id == GroundReference.id)
        .where(GroundReference.id == ground_ref_id, GroundReference.camera_id == camera_id,
               GroundModel.camera_id == camera_id, GroundModel.calibration_id == GroundReference.calibration_id)
        .order_by(GroundModel.created_at.desc(), GroundModel.id.desc()).limit(1))).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail={"code": "ground_model_not_found"})
    return GroundModelResponse.model_validate(model)


@router.post("/{ground_ref_id}/activate", response_model=GroundReferenceResponse, responses={
    409: {"description": "Activate the reference's original calibration first; a changed RTSP source requires a new reference"},
    503: {"description": "Activation persistence unavailable"},
})
async def activate_ground_reference(
    camera_id: UUID,
    ground_ref_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:write")),
):
    """Activate with the original calibration, retaining saved models and pinned jobs."""
    try:
        camera = await lock_camera(db, camera_id)
        ground_ref = (await db.execute(select(GroundReference).where(
            GroundReference.id == ground_ref_id, GroundReference.camera_id == camera_id)
            .execution_options(populate_existing=True))).scalar_one_or_none()
        if ground_ref is None:
            raise GroundReferenceError("ground_reference_not_found", 404)
        calibration = await active_calibration(db, camera_id, ground_ref.calibration_id)
        if ground_ref.source_id:
            source = await require_source(db, camera_id, ground_ref.source_id)
            if source.calibration_id != calibration.id:
                raise GroundReferenceError("ground_reference_calibration_changed")
            require_source_context(camera, calibration, source)
        model_id = await db.scalar(select(GroundModel.id).where(GroundModel.ground_ref_id == ground_ref.id,
            GroundModel.camera_id == camera_id, GroundModel.calibration_id == calibration.id).limit(1))
        if model_id is None:
            await create_ground_model_from_ref(db, camera, calibration, ground_ref,
                ppm=settings.DEFAULT_BEV_PPM, margin_m=settings.DEFAULT_BEV_MARGIN_M)
        await db.execute(update(GroundReference).where(GroundReference.camera_id == camera_id).values(is_active=False))
        ground_ref.is_active = True
        await db.commit()
        await db.refresh(ground_ref)
        return GroundReferenceResponse.model_validate(ground_ref)
    except (GroundReferenceError, ValueError) as exc:
        await db.rollback()
        raise ground_reference_error(exc) from None
    except Exception:
        await db.rollback()
        logger.error("Ground reference activation failed for camera %s", camera_id)
        raise HTTPException(status_code=503, detail={"code": "ground_reference_save_unavailable"}) from None


# Ground Model routes
model_router = APIRouter(tags=["ground-models"], dependencies=[Depends(ground_camera_access)])


@model_router.get("/active", response_model=GroundModelResponse)
async def get_active_ground_model(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """Get the ground model associated with active ground reference."""
    gm_result = await db.execute(
        select(GroundModel).join(GroundModel.ground_reference).join(Calibration, Calibration.id == GroundModel.calibration_id)
        .where(GroundModel.camera_id == camera_id, Calibration.camera_id == camera_id, Calibration.is_active == True,
               GroundReference.camera_id == camera_id, GroundReference.calibration_id == Calibration.id,
               GroundReference.is_active == True)
        .order_by(GroundReference.version.desc(), GroundModel.created_at.desc()).limit(1)
    )
    ground_model = gm_result.scalar_one_or_none()
    
    if not ground_model:
        raise HTTPException(status_code=404, detail="No compatible ground model for active calibration/reference")
    
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
    test_image_path: Optional[str] = Query(None, description="Optional raw camera-owned source key. Default uses the reference's frozen PNG, or a calibration image for legacy references."),
    mask_path: Optional[str] = Query(None, description="Relative camera-owned undistorted mask storage key with matching dimensions and its domain read scope"),
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

    ground_ref = (await db.execute(select(GroundReference).where(
        GroundReference.id == ground_model.ground_ref_id, GroundReference.camera_id == camera_id,
        GroundReference.calibration_id == calibration.id))).scalar_one_or_none()
    if ground_ref is None:
        raise HTTPException(status_code=404, detail={"code": "ground_reference_not_found"})
    source = None
    if not test_image_path and ground_ref.source_id:
        try:
            source = await require_source(db, camera_id, ground_ref.source_id)
            if source.calibration_id != calibration.id:
                raise GroundReferenceError("ground_reference_calibration_changed")
        except GroundReferenceError as exc:
            raise ground_reference_error(exc) from None
    else:
        test_image_path = test_image_path or (calibration.calib_images[0] if calibration.calib_images else None)
        if not test_image_path:
            raise HTTPException(status_code=400, detail="No test image available. Provide test_image_path.")
        await authorize_ground_image(db, test_image_path, camera_id, ground_model, calibration, api_key)
    if mask_path:
        await authorize_ground_image(db, mask_path, camera_id, ground_model, calibration, api_key, source=False)

    try:
        preview = await generate_bev_preview(
            db=db,
            camera=camera,
            calibration=calibration,
            ground_model=ground_model,
            test_image_path=test_image_path,
            mask_path=mask_path,
            source=source,
            reference_points=ground_ref.world_points,
        )
        await db.commit()
        return BEVPreviewResponse(**preview)
    except GroundReferenceError as exc:
        await db.rollback()
        raise ground_reference_error(exc) from None
    except ValueError as exc:
        if str(exc) == "bev_extent_invalid":
            raise ground_reference_error(exc) from None
        raise image_error(exc) from None
    except Exception:
        await db.rollback()
        logger.exception("BEV preview failed for ground model %s", ground_model_id)
        raise HTTPException(status_code=503, detail={"code": "bev_preview_unavailable"}) from None


# Depth Scale routes
depth_scale_router = APIRouter(tags=["depth-scales"], dependencies=[Depends(ground_camera_access)])


def image_error(exc: ValueError) -> HTTPException:
    code = str(exc)
    if code in {"depth_scale_identity_missing", "depth_scale_model_mismatch", "depth_scale_invalid_saved_fit"}:
        return HTTPException(status_code=409, detail={"code": code})
    safe = {"depth_scale_image_not_found", "prediction_invalid_image", "prediction_image_too_large",
            "prediction_resolution_mismatch", "calibration_intrinsics_missing", "calibration_invalid_intrinsics",
            "depth_scale_pixel_out_of_bounds", "depth_scale_invalid_depth", "depth_scale_degenerate_fit",
            "depth_scale_poor_fit", "depth_model_invalid_name", "depth_model_unavailable", "bev_mask_resolution_mismatch"}
    return HTTPException(status_code=404 if code == "depth_scale_image_not_found" else 503 if code == "depth_model_unavailable" else 400,
                         detail={"code": code if code in safe else "depth_scale_invalid_input"})


async def depth_context(db: AsyncSession, camera_id: UUID, ground_model_id: UUID, lock: bool = False) -> tuple[Camera, GroundModel, Calibration]:
    camera_query = select(Camera).where(Camera.id == camera_id)
    ground_query = select(GroundModel).where(GroundModel.id == ground_model_id, GroundModel.camera_id == camera_id)
    if lock:
        # Use the same camera-first lock order as calibration activation.
        camera_query = camera_query.with_for_update().execution_options(populate_existing=True)
        ground_query = ground_query.with_for_update().execution_options(populate_existing=True)
    camera = (await db.execute(camera_query)).scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail={"code": "camera_not_found"})
    ground = (await db.execute(ground_query)).scalar_one_or_none()
    if ground is None:
        raise HTTPException(status_code=404, detail={"code": "ground_model_not_found"})
    calibration = (await db.execute(select(Calibration).where(Calibration.id == ground.calibration_id, Calibration.camera_id == camera_id)
                                   .execution_options(populate_existing=True))).scalar_one_or_none()
    if calibration is None:
        raise HTTPException(status_code=404, detail={"code": "calibration_not_found"})
    return camera, ground, calibration


async def require_compatible_ground(db: AsyncSession, camera_id: UUID, ground: GroundModel, calibration: Calibration) -> None:
    selected = (await db.execute(select(GroundModel.id).join(GroundModel.ground_reference)
        .where(GroundModel.camera_id == camera_id, GroundModel.calibration_id == calibration.id,
               GroundReference.camera_id == camera_id, GroundReference.calibration_id == calibration.id,
               GroundReference.is_active == True)
        .order_by(GroundReference.version.desc(), GroundModel.created_at.desc()).limit(1))).scalar_one_or_none()
    if not calibration.is_active or selected != ground.id:
        raise HTTPException(status_code=409, detail={"code": "depth_scale_prerequisites_changed"})


async def authorize_ground_image(db: AsyncSession, key: str, camera_id: UUID, ground: GroundModel,
                                 calibration: Calibration, api_key: ApiKey, source: bool = True) -> None:
    if "\\" in key or ":" in key or any(part in {"", ".", ".."} for part in key.split("/")):
        raise HTTPException(status_code=400, detail={"code": "depth_scale_invalid_image_key"})
    if source and key in (calibration.calib_images or []):
        return
    if source and is_test_image_key(key, camera_id, ground.id):
        await require_registered_depth_image(db, key, ground.id)
        return
    parts = key.split("/")
    allowed_types = {"snapshots", "uploads", "mask", "overlay", "calibrations", "calib_images", "calib_preview"}
    if len(parts) < 4 or parts[:2] != ["cameras", str(camera_id)] or (parts[2] not in allowed_types and not parts[2].startswith("mask_")):
        raise HTTPException(status_code=404, detail={"code": "depth_scale_image_not_registered"})
    await authorize_artifact(db, key, api_key)


async def require_registered_depth_image(db: AsyncSession, key: str, ground_model_id: UUID) -> None:
    found = await db.scalar(select(PreviewArtifact.storage_key).where(
        PreviewArtifact.storage_key == key, PreviewArtifact.ground_model_id == ground_model_id))
    if found is None:
        raise HTTPException(status_code=404, detail={"code": "depth_scale_image_not_registered"})


@depth_scale_router.post("/test-images", response_model=DepthScaleImageResponse, status_code=201, responses={
    400: {"description": "Invalid/oversized image or mismatched calibration dimensions"},
    404: {"description": "Camera, ground model, or calibration missing"},
    409: {"description": "Ground model is not compatible with active calibration/reference"},
    503: {"description": "Image storage unavailable"},
})
async def upload_depth_scale_image(camera_id: UUID, ground_model_id: UUID, request: Request,
    file: UploadFile = File(..., description="Raw JPEG/PNG/BMP at calibration dimensions; legacy calibration requires an already-undistorted image"),
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("ground_ref:write")),
) -> DepthScaleImageResponse:
    camera, ground, calibration = await depth_context(db, camera_id, ground_model_id)
    await require_compatible_ground(db, camera_id, ground, calibration)
    try:
        content = await file.read(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
        image = decode_prediction_image(content)
        preview = prepare_image(image, calibration, camera)
    except ValueError as exc:
        raise image_error(exc) from None
    except Exception:
        logger.exception("Depth scale image preparation failed for ground model %s", ground_model_id)
        raise HTTPException(status_code=503, detail={"code": "depth_scale_image_unavailable"}) from None
    finally:
        await file.close()
    image_id = uuid4()
    key = test_image_key(camera_id, ground_model_id, image_id)
    try:
        storage = get_storage()
        storage.save(key, io.BytesIO(content), "application/octet-stream")
        preview_key = test_image_key(camera_id, ground_model_id, image_id, True)
        storage.save(preview_key, io.BytesIO(encode_image(preview, ".png")), "image/png")
        db.add_all([PreviewArtifact(storage_key=key, ground_model_id=ground_model_id),
                    PreviewArtifact(storage_key=preview_key, ground_model_id=ground_model_id)])
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Depth scale image storage failed for ground model %s", ground_model_id)
        raise HTTPException(status_code=503, detail={"code": "depth_scale_storage_unavailable"}) from None
    return DepthScaleImageResponse(image_id=image_id, test_image=key, width=preview.shape[1], height=preview.shape[0],
        image_url=str(request.url_for("download_depth_scale_image", camera_id=camera_id, ground_model_id=ground_model_id, image_id=image_id)))


@depth_scale_router.get("/test-images/{image_id}/image", response_class=Response, responses={
    200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}},
    404: {"description": "Image or parent ground model missing"},
    503: {"description": "Image storage unavailable"},
})
async def download_depth_scale_image(camera_id: UUID, ground_model_id: UUID, image_id: UUID,
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("ground_ref:read")),
) -> Response:
    await depth_context(db, camera_id, ground_model_id)
    await require_registered_depth_image(db, test_image_key(camera_id, ground_model_id, image_id, True), ground_model_id)
    try:
        stream = get_storage().get(test_image_key(camera_id, ground_model_id, image_id, True))
        if stream is None:
            raise HTTPException(status_code=404, detail={"code": "depth_scale_image_not_found"})
        try:
            content = stream.read(settings.CAPTURE_MAX_PIXELS * 4 + 65536)
        finally:
            stream.close()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail={"code": "depth_scale_storage_unavailable"}) from None
    return Response(content, media_type="image/png", headers={"Cache-Control": "no-store"})


# Depth preview image storage key (temporary, TTL-based)
def depth_preview_key(camera_id: UUID, ground_model_id: UUID, depth_image_id: UUID, preview: bool = False) -> str:
    filename = "depth_vis.png" if preview else "depth_raw.npy"
    return f"cameras/{camera_id}/depth-previews/{ground_model_id}/{depth_image_id}/{filename}"


async def store_depth_preview(storage, depth_raw: np.ndarray, depth_vis: np.ndarray, camera_id: UUID, ground_model_id: UUID, depth_image_id: UUID) -> None:
    """Store raw depth array and visualization for preview."""
    import io
    # Store raw depth as .npy
    raw_buf = io.BytesIO()
    np.save(raw_buf, depth_raw)
    raw_buf.seek(0)
    storage.save(depth_preview_key(camera_id, ground_model_id, depth_image_id, False), raw_buf, "application/octet-stream")
    # Store visualization as PNG
    vis_buf = io.BytesIO(encode_image(depth_vis, ".png"))
    storage.save(depth_preview_key(camera_id, ground_model_id, depth_image_id, True), vis_buf, "image/png")


async def get_depth_preview_visualization(storage, camera_id: UUID, ground_model_id: UUID, depth_image_id: UUID) -> bytes | None:
    """Get depth preview visualization."""
    stream = storage.get(depth_preview_key(camera_id, ground_model_id, depth_image_id, True))
    if stream is None:
        return None
    try:
        content = stream.read(settings.CAPTURE_MAX_PIXELS * 4 + 65536)
    finally:
        stream.close()
    return content


async def get_depth_preview_raw(storage, camera_id: UUID, ground_model_id: UUID, depth_image_id: UUID) -> np.ndarray | None:
    """Get raw depth array from preview storage."""
    import io
    stream = storage.get(depth_preview_key(camera_id, ground_model_id, depth_image_id, False))
    if stream is None:
        return None
    try:
        content = stream.read(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
    finally:
        stream.close()
    buf = io.BytesIO(content)
    return np.load(buf)


async def delete_depth_preview(storage, camera_id: UUID, ground_model_id: UUID, depth_image_id: UUID) -> None:
    """Delete depth preview files."""
    try:
        storage.delete(depth_preview_key(camera_id, ground_model_id, depth_image_id, False))
    except Exception:
        pass
    try:
        storage.delete(depth_preview_key(camera_id, ground_model_id, depth_image_id, True))
    except Exception:
        pass


@depth_scale_router.post("/preview", response_model=DepthScalePreviewResponse, responses={
    400: {"description": "Invalid image or model"},
    404: {"description": "Camera, ground model, or calibration missing"},
    409: {"description": "Ground model is not compatible with active calibration/reference"},
    503: {"description": "Model, storage, or inference unavailable/busy/timed out"},
})
async def create_depth_scale_preview(
    camera_id: UUID,
    ground_model_id: UUID,
    body: DepthScalePreviewCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("ground_ref:write")),
) -> DepthScalePreviewResponse:
    """Run YOLO depth inference on test image and return depth visualization.
    
    Returns a temporary depth_image_id and depth visualization URL.
    The raw depth array is stored temporarily (TTL ~10 min) for the subsequent create step.
    """
    camera, ground_model, calibration = await depth_context(db, camera_id, ground_model_id)
    await require_compatible_ground(db, camera_id, ground_model, calibration)
    
    test_image_key = body.test_image or (calibration.calib_images[0] if calibration.calib_images else None)
    if not test_image_key:
        raise HTTPException(status_code=400, detail={"code": "depth_scale_no_test_image", "message": "No calibration images available; upload via /test-images"})
    if test_image_key not in (calibration.calib_images or []) and not is_test_image_key(test_image_key, camera_id, ground_model_id):
        raise HTTPException(status_code=404, detail={"code": "depth_scale_image_not_registered"})
    if test_image_key not in (calibration.calib_images or []):
        await require_registered_depth_image(db, test_image_key, ground_model_id)
    
    depth_model = body.depth_model
    
    await db.commit()
    try:
        from api.services.depth_scale_service import compute_depth_preview
        result = await compute_depth_preview(
            db=db,
            camera=camera,
            calibration=calibration,
            ground_model=ground_model,
            test_image_key=test_image_key,
            depth_model=depth_model,
        )
    except ValueError as exc:
        raise image_error(exc) from None
    except VisionError as exc:
        code = "depth_scale_busy" if exc.code == "depth_scale_busy" else "depth_scale_timeout" if exc.code == "vision_timeout" else "depth_scale_inference_unavailable"
        raise HTTPException(status_code=503, detail={"code": code}) from None
    except Exception:
        logger.exception("Depth scale preview failed for ground model %s", ground_model_id)
        raise HTTPException(status_code=503, detail={"code": "depth_scale_computation_unavailable"}) from None
    
    return DepthScalePreviewResponse(
        depth_image_id=result["depth_image_id"],
        depth_visualization_url=str(request.url_for("download_depth_preview_image", camera_id=camera_id, ground_model_id=ground_model_id, depth_image_id=result["depth_image_id"])),
        width=result["width"],
        height=result["height"],
        depth_stats=result["depth_stats"],
        model_used=result["model_used"],
        imgsz=result["imgsz"],
    )


@depth_scale_router.get("/preview/{depth_image_id}/image", response_class=Response, responses={
    200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}},
    404: {"description": "Preview image not found or expired"},
    503: {"description": "Image storage unavailable"},
})
async def download_depth_preview_image(
    camera_id: UUID,
    ground_model_id: UUID,
    depth_image_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("ground_ref:read")),
) -> Response:
    """Get depth preview visualization PNG."""
    await depth_context(db, camera_id, ground_model_id)
    storage = get_storage()
    content = await get_depth_preview_visualization(storage, camera_id, ground_model_id, depth_image_id)
    if content is None:
        raise HTTPException(status_code=404, detail={"code": "depth_preview_not_found"})
    return Response(content, media_type="image/png", headers={"Cache-Control": "no-store"})


@depth_scale_router.delete("/preview/{depth_image_id}", status_code=204, responses={
    404: {"description": "Preview not found"},
    503: {"description": "Storage unavailable"},
})
async def delete_depth_preview_endpoint(
    camera_id: UUID,
    ground_model_id: UUID,
    depth_image_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("ground_ref:write")),
) -> None:
    """Delete depth preview data (cleanup)."""
    await depth_context(db, camera_id, ground_model_id)
    storage = get_storage()
    await delete_depth_preview(storage, camera_id, ground_model_id, depth_image_id)


async def persist_active_scale(db: AsyncSession, scale: DepthScale, new: bool = False) -> None:
    try:
        await db.execute(update(DepthScale).where(DepthScale.ground_model_id == scale.ground_model_id,
                         DepthScale.camera_id == scale.camera_id).values(is_active=False))
        scale.is_active = True
        if new:
            db.add(scale)
        await db.commit()
        await db.refresh(scale)
    except Exception:
        await db.rollback()
        logger.exception("Depth scale persistence failed for ground model %s", scale.ground_model_id)
        raise HTTPException(status_code=503, detail={"code": "depth_scale_persistence_unavailable"}) from None


@depth_scale_router.post("", response_model=DepthScaleResponse, status_code=status.HTTP_201_CREATED, responses={
    400: {"description": "Invalid image, measurement, depth output, or rejected fit"},
    404: {"description": "Parent resource or test image missing"},
    409: {"description": "Active prerequisites or depth model identity changed"},
    503: {"description": "Model, storage, or inference unavailable/busy/timed out"},
})
async def create_depth_scale(
    camera_id: UUID,
    ground_model_id: UUID,
    body: DepthScaleCreate,
    db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("ground_ref:write")),
):
    """Synchronously fit and auto-activate a depth correction (201), not a queued prediction.

    Requires ground_ref:write and camera access. Points use the undistorted preview;
    distance_m is camera-axis Z. Fit gates are heuristics, not metric certification.
    Rejected computations leave the previous active scale unchanged.
    
    Two modes:
    1. Legacy: Provide `test_image` (or default) + `known_distances` — runs YOLO + fits in one call
    2. Two-step: Provide `depth_image_id` (from /preview) + `known_distances` — uses stored depth from preview
    
    Only `known_distances` (min 3 points) is required.
    `test_image` defaults to first calibration image.
    `depth_model` defaults to DEFAULT_DEPTH_MODEL.
    `clip_min_m`/`clip_max_m` default to 0.1m / 50m.
    """
    camera, ground_model, calibration = await depth_context(db, camera_id, ground_model_id)
    await require_compatible_ground(db, camera_id, ground_model, calibration)
    
    # Check if using preview-based flow (new) or legacy flow
    use_preview = body.depth_image_id is not None
    
    if use_preview:
        # Two-step flow: validate depth_image_id exists in temp storage
        depth_image_id = body.depth_image_id
        depth_model = body.depth_model
        # Verify preview data exists
        from api.core.storage import get_storage
        storage = get_storage()
        stream = storage.get(depth_preview_key(camera_id, ground_model_id, depth_image_id, False))
        if stream is None:
            raise HTTPException(status_code=404, detail={"code": "depth_preview_expired", "message": "Depth preview data not found or expired. Run /preview again."})
        stream.close()
    else:
        # Legacy flow: test_image + known_distances
        test_image_key = body.test_image or (calibration.calib_images[0] if calibration.calib_images else None)
        if not test_image_key:
            raise HTTPException(status_code=400, detail={"code": "depth_scale_no_test_image", "message": "No calibration images available; upload via /test-images"})
        if test_image_key not in (calibration.calib_images or []) and not is_test_image_key(test_image_key, camera_id, ground_model_id):
            raise HTTPException(status_code=404, detail={"code": "depth_scale_image_not_registered"})
        if test_image_key not in (calibration.calib_images or []):
            await require_registered_depth_image(db, test_image_key, ground_model_id)
        depth_model = body.depth_model
    
    # Do not keep a database transaction open during bounded model execution.
    await db.commit()
    try:
        if use_preview:
            from api.services.depth_scale_service import compute_depth_scale_from_preview
            result = await compute_depth_scale_from_preview(
                db=db,
                camera=camera,
                calibration=calibration,
                ground_model=ground_model,
                depth_image_id=body.depth_image_id,
                depth_model=depth_model,
                known_distances=body.known_distances,
                clip_min_m=body.clip_min_m,
                clip_max_m=body.clip_max_m,
            )
        else:
            from api.services.depth_scale_service import compute_depth_scale
            result = await compute_depth_scale(
                db=db,
                camera=camera,
                calibration=calibration,
                ground_model=ground_model,
                test_image_key=test_image_key,
                depth_model=depth_model,
                known_distances=body.known_distances,
                clip_min_m=body.clip_min_m,
                clip_max_m=body.clip_max_m,
            )
    except ValueError as exc:
        code = str(exc)
        if code == "depth_preview_expired":
            raise HTTPException(status_code=404, detail={"code": "depth_preview_expired"}) from None
        raise image_error(exc) from None
    except VisionError as exc:
        code = "depth_scale_busy" if exc.code == "depth_scale_busy" else "depth_scale_timeout" if exc.code == "vision_timeout" else "depth_scale_inference_unavailable"
        raise HTTPException(status_code=503, detail={"code": code}) from None
    except Exception:
        logger.exception("Depth scale computation failed for ground model %s", ground_model_id)
        raise HTTPException(status_code=503, detail={"code": "depth_scale_computation_unavailable"}) from None

    camera, ground_model, calibration = await depth_context(db, camera_id, ground_model_id, lock=True)
    await require_compatible_ground(db, camera_id, ground_model, calibration)
    
    depth_scale = DepthScale(
        camera_id=camera_id,
        ground_model_id=ground_model_id,
        depth_model=result["depth_model"],
        model_identity=result["model_identity"],
        fit_provenance=result["fit_provenance"],
        scale_s=result["scale_s"],
        offset_b=result["offset_b"],
        residuals=result["residuals"],
        mean_residual_m=result["mean_residual_m"],
        max_residual_m=result["max_residual_m"],
        clip_min_m=result["clip_min_m"],
        clip_max_m=result["clip_max_m"],
        is_active=True,
    )
    try:
        validate_saved_scale(depth_scale)
        await asyncio.to_thread(validate_scale_model, depth_scale)
    except ValueError as exc:
        raise image_error(exc) from None
    await persist_active_scale(db, depth_scale, new=True)
    
    return DepthScaleResponse.model_validate(depth_scale)


@depth_scale_router.get("", response_model=DepthScaleListResponse)
async def list_depth_scales(
    camera_id: UUID,
    ground_model_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """List all depth scales for a ground model."""
    await depth_context(db, camera_id, ground_model_id)
    result = await db.execute(
        select(DepthScale)
        .where(DepthScale.ground_model_id == ground_model_id, DepthScale.camera_id == camera_id)
        .order_by(DepthScale.created_at.desc())
    )
    scales = result.scalars().all()
    return DepthScaleListResponse(
        items=[DepthScaleResponse.model_validate(s) for s in scales],
        total=len(scales),
    )


@depth_scale_router.get("/active", response_model=DepthScaleResponse, responses={
    409: {"description": "Scale identity missing, model changed, or saved fit invalid; create a new fit"},
    503: {"description": "Depth model unavailable"},
})
async def get_active_depth_scale(
    camera_id: UUID,
    ground_model_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """Get active depth scale for a ground model."""
    await depth_context(db, camera_id, ground_model_id)
    result = await db.execute(
        select(DepthScale)
        .where(
            DepthScale.ground_model_id == ground_model_id,
            DepthScale.camera_id == camera_id,
            DepthScale.is_active == True
        )
        .order_by(DepthScale.created_at.desc()).limit(1)
    )
    depth_scale = result.scalar_one_or_none()
    if not depth_scale:
        raise HTTPException(status_code=404, detail="No active depth scale for this ground model")
    try:
        validate_saved_scale(depth_scale)
        await asyncio.to_thread(validate_scale_model, depth_scale)
    except ValueError as exc:
        raise image_error(exc) from None
    return DepthScaleResponse.model_validate(depth_scale)


@depth_scale_router.get("/{depth_scale_id}", response_model=DepthScaleResponse)
async def get_depth_scale(
    camera_id: UUID,
    ground_model_id: UUID,
    depth_scale_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:read")),
):
    """Get specific depth scale."""
    await depth_context(db, camera_id, ground_model_id)
    result = await db.execute(
        select(DepthScale)
        .where(
            DepthScale.id == depth_scale_id,
            DepthScale.ground_model_id == ground_model_id,
            DepthScale.camera_id == camera_id
        )
    )
    depth_scale = result.scalar_one_or_none()
    if not depth_scale:
        raise HTTPException(status_code=404, detail="Depth scale not found")
    return DepthScaleResponse.model_validate(depth_scale)


@depth_scale_router.post("/{depth_scale_id}/activate", response_model=DepthScaleResponse, responses={
    404: {"description": "Depth scale or parent resource missing"},
    409: {"description": "Incompatible ground model or invalid saved fit; create a new scale"},
    503: {"description": "Model or persistence unavailable"},
})
async def activate_depth_scale(
    camera_id: UUID,
    ground_model_id: UUID,
    depth_scale_id: UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(require_scope("ground_ref:write")),
):
    """Set a depth scale as active (deactivates others for this ground_model)."""
    _, ground, calibration = await depth_context(db, camera_id, ground_model_id, lock=True)
    await require_compatible_ground(db, camera_id, ground, calibration)
    result = await db.execute(
        select(DepthScale)
        .where(
            DepthScale.id == depth_scale_id,
            DepthScale.ground_model_id == ground_model_id,
            DepthScale.camera_id == camera_id
        ).with_for_update().execution_options(populate_existing=True)
    )
    depth_scale = result.scalar_one_or_none()
    if not depth_scale:
        raise HTTPException(status_code=404, detail="Depth scale not found for this ground model")
    try:
        validate_saved_scale(depth_scale)
    except (ValueError, TypeError):
        raise HTTPException(status_code=409, detail={"code": "depth_scale_invalid_saved_fit"}) from None
    try:
        await asyncio.to_thread(validate_scale_model, depth_scale)
    except ValueError as exc:
        raise image_error(exc) from None
    
    await persist_active_scale(db, depth_scale)
    
    return DepthScaleResponse.model_validate(depth_scale)
