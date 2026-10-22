import asyncio
from collections.abc import Iterator
from datetime import timedelta
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.security import check_camera_access, require_scope
from api.core.storage import get_storage
from api.models.orm import ApiKey, Camera, CalibrationCaptureImage, CalibrationCaptureSession
from api.routes.streams import get_mediamtx, stream_error
from api.schemas.calibration_capture import CaptureCreate, CaptureImageList, CaptureImageResponse, CaptureResponse
from api.services.calibration_capture_service import ACTIVE_STATUSES, dispatch_capture, expire_session, finish, utcnow
from api.services.mediamtx_service import MediaMTXError, MediaMTXService, validate_rtsp_source
from api.services.calibration_compute_service import expire_computation
from api.services.snapshot_service import source_hash


router = APIRouter(tags=["calibration-captures"])


async def get_capture(db: AsyncSession, camera_id: UUID, session_id: UUID, api_key: ApiKey) -> CalibrationCaptureSession:
    check_camera_access(api_key, camera_id)
    session = await db.scalar(select(CalibrationCaptureSession).where(
        CalibrationCaptureSession.id == session_id, CalibrationCaptureSession.camera_id == camera_id,
    ).with_for_update())
    if session is None:
        raise HTTPException(status_code=404, detail={"code": "capture_session_not_found"})
    expire_session(session, utcnow())
    expire_computation(session)
    return session


@router.post("", response_model=CaptureResponse, status_code=202)
async def start_capture(
    camera_id: UUID, body: CaptureCreate,
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("calibration:write")),
    service: MediaMTXService = Depends(get_mediamtx),
) -> CaptureResponse:
    check_camera_access(api_key, camera_id)
    camera = (await db.execute(select(Camera.id, Camera.rtsp_url, Camera.stream_enabled).where(
        Camera.id == camera_id,
    ).with_for_update())).one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail={"code": "camera_not_found"})
    if not camera.stream_enabled or not camera.rtsp_url:
        raise HTTPException(status_code=409, detail={"code": "capture_requires_enabled_stream"})
    try:
        service.check_config()
        service.check_config(read=True)
        validate_rtsp_source(camera.rtsp_url, service.config)
    except MediaMTXError as exc:
        raise stream_error(exc) from None
    active = await db.scalar(select(CalibrationCaptureSession).where(
        CalibrationCaptureSession.camera_id == camera_id,
        CalibrationCaptureSession.status.in_(ACTIVE_STATUSES),
    ).with_for_update())
    now = utcnow()
    if active is not None and not expire_session(active, now):
        raise HTTPException(status_code=409, detail={"code": "capture_already_active", "session_id": str(active.id)})
    await db.flush()
    session = CalibrationCaptureSession(
        id=uuid4(), camera_id=camera_id, status="queued", created_at=now,
        deadline_at=now + timedelta(seconds=body.max_duration_sec),
        attempted=0, captured=0, failed_attempts=0,
        selection_version=1, accepted=0, rejected=0, rejection_counts={}, compute_status="none",
        source_hash=source_hash(camera.rtsp_url),
        **body.model_dump(),
    )
    db.add(session)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail={"code": "capture_already_active"}) from None
    except Exception:
        await db.rollback()
        raise HTTPException(status_code=503, detail={"code": "capture_database_unavailable"}) from None
    response = CaptureResponse.model_validate(session)
    await dispatch_capture(db, session.id)
    return response


@router.get("/{session_id}", response_model=CaptureResponse)
async def capture_status(
    camera_id: UUID, session_id: UUID, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("calibration:read")),
) -> CaptureResponse:
    session = await get_capture(db, camera_id, session_id, api_key)
    await db.commit()
    return CaptureResponse.model_validate(session)


@router.post("/{session_id}/stop", response_model=CaptureResponse)
async def stop_capture(
    camera_id: UUID, session_id: UUID, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("calibration:write")),
) -> CaptureResponse:
    session = await get_capture(db, camera_id, session_id, api_key)
    if session.status == "queued":
        finish(session, "stopped", "user_stop", utcnow())
    elif session.status == "running":
        session.status = "stopping"
    await db.commit()
    return CaptureResponse.model_validate(session)


@router.get("/{session_id}/images", response_model=CaptureImageList)
async def capture_images(
    camera_id: UUID, session_id: UUID, request: Request,
    offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("calibration:read")),
) -> CaptureImageList:
    await get_capture(db, camera_id, session_id, api_key)
    await db.commit()
    images = (await db.execute(select(CalibrationCaptureImage).where(
        CalibrationCaptureImage.session_id == session_id,
    ).order_by(CalibrationCaptureImage.sequence).offset(offset).limit(limit))).scalars().all()
    total = await db.scalar(select(func.count()).select_from(CalibrationCaptureImage).where(
        CalibrationCaptureImage.session_id == session_id,
    ))
    return CaptureImageList(items=[CaptureImageResponse(
        id=image.id, sequence=image.sequence, width=image.width, height=image.height,
        size_bytes=image.size_bytes, captured_at=image.captured_at,
        download_url=str(request.url_for("download_capture_image", camera_id=camera_id, session_id=session_id, image_id=image.id)),
        quality_status=image.quality_status, rejection_reason=image.rejection_reason,
        sharpness=image.sharpness, corners=image.corners,
        preview_url=str(request.url_for("preview_capture_image", camera_id=camera_id, session_id=session_id, image_id=image.id)),
    ) for image in images], total=total)


async def image_response(storage_key: str) -> StreamingResponse:
    try:
        stream = await asyncio.to_thread(lambda: get_storage().get(storage_key))
    except Exception:
        raise HTTPException(status_code=503, detail={"code": "capture_storage_unavailable"}) from None
    if stream is None:
        raise HTTPException(status_code=404, detail={"code": "capture_image_missing"})

    def chunks() -> Iterator[bytes]:
        try:
            while chunk := stream.read(65536):
                yield chunk
        finally:
            stream.close()

    return StreamingResponse(chunks(), media_type="image/jpeg", headers={
        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
    })


@router.get("/{session_id}/images/{image_id}", response_class=StreamingResponse, responses={200: {"content": {"image/jpeg": {}}}})
async def download_capture_image(
    camera_id: UUID, session_id: UUID, image_id: UUID, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("calibration:read")),
) -> StreamingResponse:
    await get_capture(db, camera_id, session_id, api_key)
    await db.commit()
    image = await db.scalar(select(CalibrationCaptureImage).where(
        CalibrationCaptureImage.id == image_id, CalibrationCaptureImage.session_id == session_id,
    ))
    if image is None:
        raise HTTPException(status_code=404, detail={"code": "capture_image_not_found"})
    return await image_response(image.storage_key)


@router.get("/{session_id}/images/{image_id}/preview", response_class=StreamingResponse, responses={200: {"content": {"image/jpeg": {}}}})
async def preview_capture_image(
    camera_id: UUID, session_id: UUID, image_id: UUID, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("calibration:read")),
) -> StreamingResponse:
    from io import BytesIO
    from api.services.calibration_quality_service import corner_preview
    from api.services.rtsp_capture_service import decode_frame
    session = await get_capture(db, camera_id, session_id, api_key)
    cols, rows = session.chessboard_cols, session.chessboard_rows
    await db.commit()
    image = await db.scalar(select(CalibrationCaptureImage).where(CalibrationCaptureImage.id == image_id, CalibrationCaptureImage.session_id == session_id))
    if image is None:
        raise HTTPException(status_code=404, detail={"code": "capture_image_not_found"})
    def render() -> bytes:
        from api.core.config import settings
        stream = get_storage().get(image.storage_key)
        if stream is None:
            raise ValueError("capture_image_missing")
        try:
            content = stream.read(settings.CAPTURE_MAX_FRAME_BYTES + 1)
        finally:
            stream.close()
        decode_frame(content)
        return corner_preview(content, image.corners, cols, rows)
    try:
        content = await asyncio.to_thread(render)
    except Exception:
        raise HTTPException(status_code=503, detail={"code": "capture_preview_unavailable"}) from None
    return StreamingResponse(BytesIO(content), media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


@router.post("/{session_id}/compute", response_model=CaptureResponse, status_code=202)
async def compute_capture(
    camera_id: UUID, session_id: UUID, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("calibration:write")),
) -> CaptureResponse:
    from api.core.config import settings
    session = await get_capture(db, camera_id, session_id, api_key)
    if session.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail={"code": "capture_not_finished"})
    available = session.accepted if session.selection_version else session.captured
    if available < settings.MIN_CALIB_IMAGES:
        raise HTTPException(status_code=409, detail={"code": "calibration_insufficient_images"})
    if session.compute_status not in {"queued", "running", "done"}:
        session.compute_status = "queued"
        session.compute_error_code = None
        session.compute_requested_at = utcnow()
        session.compute_dispatched_at = None
        session.compute_started_at = session.compute_completed_at = session.compute_token = None
    await db.commit()
    return CaptureResponse.model_validate(session)


async def authorize_capture_artifact(db: AsyncSession, key: str, api_key: ApiKey) -> None:
    if "*" not in api_key.scopes and "calibration:read" not in api_key.scopes:
        raise HTTPException(status_code=403, detail="Scope 'calibration:read' required")
    camera_id = await db.scalar(select(CalibrationCaptureSession.camera_id).join(
        CalibrationCaptureImage, CalibrationCaptureImage.session_id == CalibrationCaptureSession.id,
    ).where(CalibrationCaptureImage.storage_key == key))
    if camera_id is None:
        raise HTTPException(status_code=404, detail={"code": "capture_image_not_found"})
    check_camera_access(api_key, camera_id)
