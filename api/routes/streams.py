import hmac
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import httpx

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.security import check_camera_access, hash_key, require_scope
from api.models.orm import ApiKey, Camera, StreamSession
from api.schemas.stream import StreamResponse, StreamUpdate
from api.services.mediamtx_service import MediaMTXError, MediaMTXService, desired_path, stream_path, validate_rtsp_source
from api.services.stream_service import reconcile_streams


router = APIRouter(tags=["streams"])


def get_mediamtx(request: Request) -> MediaMTXService:
    return request.app.state.mediamtx


def stream_error(exc: MediaMTXError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail={"code": exc.code})


async def get_camera(db: AsyncSession, camera_id: UUID, api_key: ApiKey) -> Camera:
    check_camera_access(api_key, camera_id)
    camera = (await db.execute(select(Camera).where(Camera.id == camera_id))).scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


def response_state(camera: Camera, request: Request) -> StreamResponse:
    return StreamResponse(
        camera_id=camera.id, enabled=camera.stream_enabled,
        status="unknown" if camera.stream_enabled else "disabled", synced=False,
        whep_url=str(request.url_for("whep_offer", camera_id=str(camera.id))) if camera.stream_enabled else None,
        checked_at=datetime.now(timezone.utc),
    )


@router.get("/stream", response_model=StreamResponse)
async def get_stream(
    camera_id: UUID, request: Request, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")),
    service: MediaMTXService = Depends(get_mediamtx),
) -> StreamResponse:
    camera = await get_camera(db, camera_id, api_key)
    state = response_state(camera, request)
    try:
        config = await service.get_path_config(stream_path(camera_id))
        if not camera.stream_enabled:
            state.synced = config is None
            return state
        validate_rtsp_source(camera.rtsp_url or "", service.config)
        state.synced = config is not None and all(config.get(k) == v for k, v in desired_path(camera.rtsp_url).items())
        if state.synced:
            runtime = await service.runtime_path(stream_path(camera_id))
            if runtime is not None:
                state.tracks = [track["codec"] for track in runtime.get("tracks2", []) if isinstance(track, dict) and isinstance(track.get("codec"), str)]
                # v1.20.0 marks an initialized static source online before it has frames.
                if runtime.get("online") and runtime.get("available"):
                    state.status = "online"
                elif runtime.get("readers") or runtime.get("available"):
                    state.status = "offline"
                else:
                    state.status = "idle"
    except MediaMTXError:
        state.synced = False
    return state


@router.patch("/stream", response_model=StreamResponse)
async def update_stream(
    camera_id: UUID, data: StreamUpdate, request: Request, background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db), api_key: ApiKey = Depends(require_scope("camera:write")),
    service: MediaMTXService = Depends(get_mediamtx),
) -> StreamResponse:
    camera = await get_camera(db, camera_id, api_key)
    if data.enabled:
        try:
            service.check_config()
            service.check_config(read=True)
            validate_rtsp_source(camera.rtsp_url or "", service.config)
        except MediaMTXError as exc:
            raise stream_error(exc) from None
    camera.stream_enabled = data.enabled
    await db.commit()
    background_tasks.add_task(reconcile_streams, service)
    return response_state(camera, request)


async def playable_camera(db: AsyncSession, camera_id: UUID, api_key: ApiKey, service: MediaMTXService) -> Camera:
    camera = await get_camera(db, camera_id, api_key)
    if not camera.stream_enabled:
        raise HTTPException(status_code=409, detail={"code": "stream_disabled"})
    try:
        validate_rtsp_source(camera.rtsp_url or "", service.config)
        config = await service.get_path_config(stream_path(camera_id))
        if config is None or any(config.get(k) != v for k, v in desired_path(camera.rtsp_url).items()):
            raise MediaMTXError("stream_sync_pending")
    except MediaMTXError as exc:
        raise stream_error(exc) from None
    return camera


def signaling_response(upstream: httpx.Response, *, location: str | None = None) -> Response:
    headers = {"Cache-Control": "no-store"}
    for name in ("Content-Type", "ETag", "Accept-Post", "Accept-Patch", "Link"):
        values = upstream.headers.get_list(name)
        if values:
            headers[name] = ", ".join(values)
    if location is not None:
        headers["Location"] = location
    return Response(content=upstream.content if upstream.status_code != 204 else b"", status_code=upstream.status_code, headers=headers)


async def read_sdp(request: Request, content_type: str, service: MediaMTXService) -> bytes:
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != content_type:
        raise HTTPException(status_code=415, detail={"code": "invalid_signaling_content_type"})
    body = bytearray()
    try:
        async with asyncio.timeout(service.config.STREAM_TIMEOUT):
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > service.config.MEDIAMTX_MAX_SDP_BYTES:
                    raise HTTPException(status_code=413, detail={"code": "sdp_too_large"})
    except TimeoutError:
        raise HTTPException(status_code=408, detail={"code": "sdp_upload_timeout"}) from None
    if not body:
        raise HTTPException(status_code=400, detail={"code": "empty_sdp"})
    return bytes(body)


@router.options("/stream/whep", status_code=204, responses={
    204: {"description": "Authenticated WHEP capabilities (not a CORS preflight)", "headers": {
        "Accept-Post": {"schema": {"type": "string"}},
        "Link": {"description": "ICE server links when returned by MediaMTX", "schema": {"type": "string"}},
    }},
})
async def whep_options(
    camera_id: UUID, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")), service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    await playable_camera(db, camera_id, api_key, service)
    try:
        return signaling_response(await service.signaling("OPTIONS", camera_id))
    except MediaMTXError as exc:
        raise stream_error(exc) from None


SDP_BODY = {"requestBody": {"required": True, "content": {"application/sdp": {"schema": {"type": "string"}}}}}
SDP_RESPONSE = {201: {"description": "SDP answer and owned backend session Location; do not retry offer POST automatically", "content": {"application/sdp": {"schema": {"type": "string"}}}, "headers": {"Location": {"description": "Backend URL for authenticated PATCH/DELETE with the same API key", "schema": {"type": "string"}}, "ETag": {"schema": {"type": "string"}}, "Link": {"schema": {"type": "string"}}}}}


@router.post("/stream/whep", status_code=201, response_class=Response, responses=SDP_RESPONSE, openapi_extra=SDP_BODY)
async def whep_offer(
    camera_id: UUID, request: Request, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")), service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    await playable_camera(db, camera_id, api_key, service)
    body = await read_sdp(request, "application/sdp", service)
    try:
        upstream = await service.signaling("POST", camera_id, content=body, headers={"Content-Type": "application/sdp"})
        upstream_id = service.parse_session_location(upstream.headers.get("location", ""), camera_id)
    except MediaMTXError as exc:
        raise stream_error(exc) from None
    session = StreamSession(
        id=uuid4(), camera_id=camera_id, owner_hash=hash_key(request.headers["X-API-Key"]),
        upstream_id=upstream_id,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=service.config.MEDIAMTX_SESSION_TTL_SEC),
    )
    try:
        db.add(session)
        await db.commit()
    except Exception:
        await db.rollback()
        try:
            await service.signaling("DELETE", camera_id, upstream_id)
        except MediaMTXError:
            pass
        raise HTTPException(status_code=503, detail={"code": "stream_session_storage_unavailable"}) from None
    location = str(request.url_for("whep_patch", camera_id=str(camera_id), session_id=str(session.id)))
    return signaling_response(upstream, location=location)


async def owned_session(db: AsyncSession, camera_id: UUID, session_id: UUID, request: Request, api_key: ApiKey) -> StreamSession:
    check_camera_access(api_key, camera_id)
    session = (await db.execute(select(StreamSession).where(
        StreamSession.id == session_id, StreamSession.camera_id == camera_id,
        StreamSession.expires_at > datetime.now(timezone.utc),
    ))).scalar_one_or_none()
    if session is None or not hmac.compare_digest(session.owner_hash, hash_key(request.headers["X-API-Key"])):
        raise HTTPException(status_code=404, detail={"code": "stream_session_not_found"})
    return session


@router.patch("/stream/whep/{session_id}", status_code=204, response_class=Response,
    responses={200: {"description": "ICE restart answer", "content": {"application/trickle-ice-sdpfrag": {"schema": {"type": "string"}}}, "headers": {"ETag": {"schema": {"type": "string"}}}},
               204: {"description": "ICE fragment accepted", "headers": {"ETag": {"schema": {"type": "string"}}}}},
    openapi_extra={"requestBody": {"required": True, "content": {"application/trickle-ice-sdpfrag": {"schema": {"type": "string"}}}},
                   "parameters": [{"name": "If-Match", "in": "header", "required": False, "description": "ETag from the WHEP answer; forwarded when present", "schema": {"type": "string"}}]})
async def whep_patch(
    camera_id: UUID, session_id: UUID, request: Request, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")), service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    session = await owned_session(db, camera_id, session_id, request, api_key)
    await playable_camera(db, camera_id, api_key, service)
    body = await read_sdp(request, "application/trickle-ice-sdpfrag", service)
    headers = {"Content-Type": "application/trickle-ice-sdpfrag"}
    if "if-match" in request.headers:
        headers["If-Match"] = request.headers["if-match"]
    try:
        return signaling_response(await service.signaling("PATCH", camera_id, session.upstream_id, content=body, headers=headers))
    except MediaMTXError as exc:
        if exc.status_code == 404:
            await db.delete(session)
            await db.commit()
        raise stream_error(exc) from None


@router.delete("/stream/whep/{session_id}", status_code=204)
async def whep_delete(
    camera_id: UUID, session_id: UUID, request: Request, db: AsyncSession = Depends(get_db),
    api_key: ApiKey = Depends(require_scope("camera:read")), service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    session = await owned_session(db, camera_id, session_id, request, api_key)
    try:
        await service.signaling("DELETE", camera_id, session.upstream_id)
    except MediaMTXError as exc:
        raise stream_error(exc) from None
    await db.delete(session)
    await db.commit()
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
