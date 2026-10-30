import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.security import check_camera_access, require_scope
from api.core.storage import get_storage
from api.models.orm import ApiKey, GroundReferenceSource
from api.schemas.ground_ref import GroundReferenceSourceCreate, GroundReferenceSourceResponse
from api.services.ground_reference_source_service import GroundReferenceError, prepare_source, read_source_bytes, require_source


router = APIRouter(tags=["ground-reference-sources"])
logger = logging.getLogger(__name__)


def ground_reference_error(exc: GroundReferenceError | ValueError) -> HTTPException:
    if isinstance(exc, GroundReferenceError):
        return HTTPException(status_code=exc.status, detail={"code": exc.code})
    code = str(exc)
    safe = {"prediction_invalid_image", "prediction_image_too_large", "prediction_resolution_mismatch",
            "calibration_intrinsics_missing", "calibration_invalid_intrinsics", "ground_reference_invalid_points",
            "ground_reference_duplicate_points", "ground_reference_degenerate_points", "ground_reference_pixel_out_of_bounds",
            "ground_reference_invalid_geometry", "bev_extent_invalid"}
    return HTTPException(status_code=400, detail={"code": code if code in safe else "ground_reference_invalid_input"})


def source_response(source: GroundReferenceSource, request: Request) -> GroundReferenceSourceResponse:
    return GroundReferenceSourceResponse(
        id=source.id, camera_id=source.camera_id, snapshot_id=source.snapshot_id,
        calibration_id=source.calibration_id, captured_at=source.captured_at, width=source.width, height=source.height,
        image_url=str(request.url_for("download_ground_reference_source", camera_id=source.camera_id, source_id=source.id)))


@router.post("", response_model=GroundReferenceSourceResponse, status_code=201, responses={
    200: {"model": GroundReferenceSourceResponse, "description": "Same snapshot/calibration/preparation reused"},
    400: {"description": "Invalid/oversized source image or intrinsics"},
    404: {"description": "Camera, snapshot or source image missing"},
    409: {"description": "Active calibration, RTSP source or dimensions changed; acquire a new source"},
    503: {"description": "Source image preparation, storage or persistence unavailable"},
}, dependencies=[Depends(require_scope("camera:read")), Depends(require_scope("calibration:read"))])
async def create_ground_reference_source(camera_id: UUID, body: GroundReferenceSourceCreate,
    request: Request, response: Response, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("ground_ref:write")),
) -> GroundReferenceSourceResponse:
    """Freeze an original RTSP snapshot in undistorted pixels using the active calibration.

    Requires ground_ref:write, camera:read and calibration:read. Image bytes and provenance
    commit before the response. Repeating the same preparation is idempotent.
    """
    check_camera_access(api_key, camera_id)
    try:
        source, created = await prepare_source(db, camera_id, body.snapshot_id, body.calibration_id, get_storage())
        await db.commit()
        response.status_code = 201 if created else 200
        return source_response(source, request)
    except (GroundReferenceError, ValueError) as exc:
        await db.rollback()
        raise ground_reference_error(exc) from None
    except Exception:
        await db.rollback()
        logger.error("Ground reference source preparation failed for camera %s", camera_id)
        raise HTTPException(status_code=503, detail={"code": "ground_reference_source_unavailable"}) from None


@router.get("/{source_id}", response_model=GroundReferenceSourceResponse, responses={404: {"description": "Source not found for camera"}})
async def get_ground_reference_source(camera_id: UUID, source_id: UUID, request: Request,
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("ground_ref:read")),
) -> GroundReferenceSourceResponse:
    check_camera_access(api_key, camera_id)
    try:
        return source_response(await require_source(db, camera_id, source_id), request)
    except GroundReferenceError as exc:
        raise ground_reference_error(exc) from None


@router.get("/{source_id}/image", response_class=Response, responses={
    200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}},
    404: {"description": "Source or registered image missing"},
    409: {"description": "Stored source image failed integrity check"},
    503: {"description": "Image storage unavailable"},
})
async def download_ground_reference_source(camera_id: UUID, source_id: UUID,
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("ground_ref:read")),
) -> Response:
    check_camera_access(api_key, camera_id)
    try:
        source = await require_source(db, camera_id, source_id)
        content = await asyncio.to_thread(read_source_bytes, source, get_storage())
    except GroundReferenceError as exc:
        raise ground_reference_error(exc) from None
    except Exception:
        raise HTTPException(status_code=503, detail={"code": "ground_reference_source_unavailable"}) from None
    return Response(content, media_type="image/png", headers={"Cache-Control": "no-store"})
