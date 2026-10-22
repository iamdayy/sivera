from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db, get_db_context
from api.core.security import check_camera_access, require_scope
from api.core.storage import get_storage
from api.models.orm import ApiKey, Camera, CameraSnapshot
from api.routes.calibration_captures import image_response
from api.schemas.snapshot import SnapshotResponse
from api.services.rtsp_capture_service import CaptureError
from api.services.snapshot_service import capture_snapshot, latest_snapshot, require_rtsp_camera


router = APIRouter(tags=["snapshots"])


def snapshot_error(exc: CaptureError) -> HTTPException:
    code = exc.code
    status = 404 if code in {"camera_not_found", "snapshot_not_found"} else 409 if code in {
        "snapshot_stale", "snapshot_source_changed", "snapshot_requires_enabled_stream",
    } else 503
    return HTTPException(status_code=status, detail={"code": code})


def response(snapshot: CameraSnapshot, request: Request) -> SnapshotResponse:
    return SnapshotResponse(
        id=snapshot.id, camera_id=snapshot.camera_id, width=snapshot.width, height=snapshot.height,
        size_bytes=snapshot.size_bytes, captured_at=snapshot.captured_at,
        image_url=str(request.url_for("download_snapshot", camera_id=snapshot.camera_id, snapshot_id=snapshot.id)),
    )


@router.post("", response_model=SnapshotResponse, status_code=201)
async def create_snapshot(camera_id: UUID, request: Request, api_key: ApiKey = Depends(require_scope("camera:write"))) -> SnapshotResponse:
    check_camera_access(api_key, camera_id)
    try:
        snapshot = await capture_snapshot(camera_id, get_db_context, get_storage())
    except CaptureError as exc:
        raise snapshot_error(exc) from None
    except Exception:
        raise HTTPException(status_code=503, detail={"code": "snapshot_unavailable"}) from None
    return response(snapshot, request)


@router.get("/latest", response_model=SnapshotResponse)
async def get_latest_snapshot(
    camera_id: UUID, request: Request, max_frame_age_sec: int = Query(30, ge=1, le=300),
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("camera:read")),
) -> SnapshotResponse:
    check_camera_access(api_key, camera_id)
    camera = (await db.execute(select(Camera.rtsp_url, Camera.stream_enabled).where(Camera.id == camera_id))).one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail={"code": "camera_not_found"})
    try:
        snapshot = await latest_snapshot(db, camera_id, require_rtsp_camera(camera), max_frame_age_sec)
    except CaptureError as exc:
        raise snapshot_error(exc) from None
    return response(snapshot, request)


@router.get("/{snapshot_id}/image", response_class=StreamingResponse, responses={200: {"content": {"image/jpeg": {}}}})
async def download_snapshot(
    camera_id: UUID, snapshot_id: UUID, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")),
) -> StreamingResponse:
    check_camera_access(api_key, camera_id)
    snapshot = await db.scalar(select(CameraSnapshot).where(CameraSnapshot.id == snapshot_id, CameraSnapshot.camera_id == camera_id))
    if snapshot is None:
        raise HTTPException(status_code=404, detail={"code": "snapshot_not_found"})
    return await image_response(snapshot.storage_key)
