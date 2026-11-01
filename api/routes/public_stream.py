"""Public WHEP signaling for enabled cameras."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.models.orm import Camera
from api.routes.streams import get_mediamtx, read_sdp, signaling_response, stream_error
from api.services.mediamtx_service import MediaMTXError, MediaMTXService, stream_path


router = APIRouter(tags=["public-stream"])


async def get_camera_or_404(db: AsyncSession, camera_id: UUID) -> Camera:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


async def _forward_whep_request(
    camera_id: UUID,
    method: str,
    request: Request,
    service: MediaMTXService,
    session_id: UUID | None = None,
) -> Response:
    headers = {}
    body = None
    if method in {"POST", "PATCH"}:
        content_type = "application/sdp" if method == "POST" else "application/trickle-ice-sdpfrag"
        body = await read_sdp(request, content_type, service)
        headers["Content-Type"] = content_type
    if "if-match" in request.headers:
        headers["If-Match"] = request.headers["if-match"]

    try:
        upstream = await service.signaling(method, camera_id, session_id, content=body, headers=headers)
        location = None
        if method == "POST":
            upstream_id = service.parse_session_location(upstream.headers.get("location", ""), camera_id)
            location = str(request.url_for("whep_patch_session", camera_id=str(camera_id), session_id=str(upstream_id)))
    except MediaMTXError as exc:
        raise stream_error(exc) from None

    # MediaMTX also returns 200 on DELETE; the public WHEP contract is bodyless 204.
    if method == "DELETE":
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return signaling_response(upstream, location=location)


@router.options("/cameras/{camera_id}/stream/whep", status_code=204, response_class=Response, responses={
    204: {"description": "Public WHEP capabilities (not a CORS preflight)", "headers": {
        "Accept-Post": {"schema": {"type": "string"}},
        "Accept-Patch": {"schema": {"type": "string"}},
        "Link": {"description": "ICE server links when returned by MediaMTX", "schema": {"type": "string"}},
    }},
})
async def whep_options(
    camera_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    """Read capabilities from MediaMTX without an API key."""
    camera = await get_camera_or_404(db, camera_id)
    if not camera.stream_enabled:
        raise HTTPException(status_code=403, detail="Stream not enabled")
    return await _forward_whep_request(camera_id, "OPTIONS", request, service)


@router.get("/cameras/{camera_id}/stream/whep")
async def whep_get_offer(
    camera_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    """Forward legacy GET requests; players establish sessions with POST."""
    camera = await get_camera_or_404(db, camera_id)
    if not camera.stream_enabled:
        raise HTTPException(status_code=403, detail="Stream not enabled")

    try:
        upstream = await service.request("GET", f"/{stream_path(camera_id)}/whep", read=True)
        service.require_success(upstream)
        return signaling_response(upstream)
    except MediaMTXError as exc:
        raise stream_error(exc) from None


@router.post("/cameras/{camera_id}/stream/whep", status_code=201, response_class=Response, responses={
    201: {"description": "SDP answer and public backend session Location; do not retry offer POST automatically",
          "content": {"application/sdp": {"schema": {"type": "string"}}},
          "headers": {"Location": {"description": "Backend URL for public PATCH/DELETE", "schema": {"type": "string"}},
                      "ETag": {"schema": {"type": "string"}}, "Link": {"schema": {"type": "string"}}}},
}, openapi_extra={"requestBody": {"required": True, "content": {"application/sdp": {"schema": {"type": "string"}}}}})
async def whep_post_answer(
    camera_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    """Send the browser's SDP offer and return MediaMTX's answer."""
    camera = await get_camera_or_404(db, camera_id)
    if not camera.stream_enabled:
        raise HTTPException(status_code=403, detail="Stream not enabled")

    return await _forward_whep_request(camera_id, "POST", request, service)


@router.patch("/cameras/{camera_id}/stream/whep/{session_id}", status_code=204, response_class=Response,
    responses={200: {"description": "ICE restart answer", "content": {"application/trickle-ice-sdpfrag": {"schema": {"type": "string"}}}, "headers": {"ETag": {"schema": {"type": "string"}}}},
               204: {"description": "ICE fragment accepted", "headers": {"ETag": {"schema": {"type": "string"}}}}},
    openapi_extra={"requestBody": {"required": True, "content": {"application/trickle-ice-sdpfrag": {"schema": {"type": "string"}}}},
                   "parameters": [{"name": "If-Match", "in": "header", "required": False, "description": "ETag from the WHEP answer", "schema": {"type": "string"}}]})
async def whep_patch_session(
    camera_id: UUID,
    session_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    """WHEP PATCH: ICE restart or session update.

    No API key required. Camera must have stream_enabled=true.
    """
    camera = await get_camera_or_404(db, camera_id)
    if not camera.stream_enabled:
        raise HTTPException(status_code=403, detail="Stream not enabled")

    return await _forward_whep_request(camera_id, "PATCH", request, service, session_id)


@router.delete("/cameras/{camera_id}/stream/whep/{session_id}", status_code=204, response_class=Response)
async def whep_delete_session(
    camera_id: UUID,
    session_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    service: MediaMTXService = Depends(get_mediamtx),
) -> Response:
    """WHEP DELETE: terminate WebRTC session.

    No API key required. Camera must have stream_enabled=true.
    """
    camera = await get_camera_or_404(db, camera_id)
    if not camera.stream_enabled:
        raise HTTPException(status_code=403, detail="Stream not enabled")

    return await _forward_whep_request(camera_id, "DELETE", request, service, session_id)
