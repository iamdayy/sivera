"""FastAPI application factory and main entry point."""
import mimetypes
import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.openapi.utils import get_openapi
from sqlalchemy import text

from api.core.config import settings, STORAGE_ROOT
from api.core.database import AsyncSession, get_db, init_db, close_db, engine
from api.core.security import verify_api_key
from api.core.storage import get_storage
from api.routes import cameras, calibrations, calibration_captures, ground_refs, predictions, streams, snapshots
from api.schemas.system import HealthResponse, RootResponse
from api.services.mediamtx_service import MediaMTXService
from api.services.stream_service import reconciliation_loop
from api.services.calibration_capture_service import capture_reconciliation_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    await init_db()
    if settings.STORAGE_BACKEND.lower() == "local":
        # Fail startup rather than let API and workers silently use different roots.
        STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    app.state.mediamtx = MediaMTXService()
    reconciler = asyncio.create_task(reconciliation_loop(app.state.mediamtx))
    capture_reconciler = asyncio.create_task(capture_reconciliation_loop())
    try:
        yield
    finally:
        reconciler.cancel()
        with suppress(asyncio.CancelledError):
            await reconciler
        capture_reconciler.cancel()
        with suppress(asyncio.CancelledError):
            await capture_reconciler
        await app.state.mediamtx.close()
        await close_db()


# Custom OpenAPI schema with security
def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "## Waste Volume Estimation API\n\n"
            "Camera operations are mounted at `/api/v1/cameras`; `/files` serves protected artifacts.\n\n"
            "- WebRTC playback uses backend-authenticated WHEP signaling; media flows directly from MediaMTX.\n"
            "- RTSP calibration capture selects chessboard frames and queues computation. "
            "Computed calibrations are inactive until explicitly activated; multipart calibration upload "
            "is synchronous and activates its result.\n"
            "- Snapshots preserve original resolution and record worker-reception UTC time, not sensor time.\n"
            "- Predictions queue `upload`, `rtsp_frame`, or `latest_rtsp` and persist "
            "`pending -> processing -> done/failed`. HTTP 202 acknowledges submission, not completion.\n"
            "- Grid volume sums cell median heights times cell area. `convex_hull` means "
            "2D footprint-hull area times median height, not a 3D hull. "
            "Volume is m³/L, footprint m², height meters; missing estimates remain null. "
            "BEV is a visualization, not a complete 3D reconstruction.\n\n"
            "### Authentication\n"
            "Camera operations and artifact downloads require `X-API-Key`. "
            "`/`, `/health`, `/docs`, `/redoc`, and `/openapi.json` are public. "
            "Keys are configured through `VALID_API_KEYS` or stored hashed in PostgreSQL. "
            "Environment-configured keys have wildcard scopes and must not be embedded in browser builds. "
            "Missing headers currently return 422; invalid keys return 401 and insufficient scopes return 403.\n\n"
            "### Scopes\n"
            "- `camera:read` / `camera:write` - Camera management\n"
            "- `calibration:read` / `calibration:write` - Calibration\n"
            "- `ground_ref:read` / `ground_ref:write` - Ground references\n"
            "- `predict:read` / `predict:write` - Predictions\n\n"
            "### Prediction prerequisites and coordinates\n"
            "Submission pins an active calibration, a ground model linked to an active reference "
            "for that calibration, and an active depth scale for that ground model. "
            "There is no public depth-scale setup/readiness endpoint; provisioning is an operator task. "
            "New calibrations preserve `k_original`, `k_undist`, and dimensions; `k_matrix` "
            "remains the undistorted K. Upload raw images for these calibrations. Legacy uploads "
            "without original intrinsics require already-undistorted images; RTSP requires recalibration. "
            "Ground-reference image points use undistorted coordinates.\n\n"
            "### RTSP sources and output selection\n"
            "`rtsp_frame` captures at worker execution. `latest_rtsp` requires a fresh persisted "
            "snapshot at submission and selects again at execution, without fallback capture. "
            "RTSP rejects caller files/timestamps; frame dimensions must exactly match calibration. "
            "`return_outputs` is a JSON-array form string: omitted registers all supported outputs, "
            "`[]` registers none. Internal files may still be generated; summary/instances/volumes "
            "are not filtered. Broker failure returns 503 and marks the job failed; predictions "
            "are not automatically redispatched.\n\n"
            "### Errors and deployment limits\n"
            "Errors use `detail` as a string, an object with a safe `code`, or a validation list. "
            "Handle all shapes. Browser video/ICE/reconnect, real-camera accuracy, model/GPU "
            "inference, TLS/TURN, and dashboard integration still require deployment testing. "
            "Legacy ground-reference/model routes do not yet enforce camera-bound API-key restrictions; "
            "do not treat them as tenant-isolated."
        ),
        routes=app.routes,
        tags=[
            {"name": "cameras", "description": "Camera registry and management"},
            {"name": "streams", "description": "Camera stream state and authenticated WebRTC/WHEP signaling"},
            {"name": "calibrations", "description": "calibration:read/write. Synchronous multipart upload auto-activates; capture results require explicit activation. k_matrix is undistorted K."},
            {"name": "calibration-captures", "description": "calibration:read/write, camera-bound keys. PostgreSQL owns capture and computation progress; accepted-frame target, 300 stored frames maximum."},
            {"name": "snapshots", "description": "camera:write for synchronous capture, camera:read for latest/image. Original-resolution RTSP snapshots; reception timestamps, default freshness 30 seconds."},
            {"name": "ground-references", "description": "Ground plane reference points and homography"},
            {"name": "ground-models", "description": "Computed ground models and BEV preview"},
            {"name": "predictions", "description": "predict:write for queued submission; predict:read for status/history/outputs/instances/volumes. Compatible active calibration, ground reference/model, and depth scale required."},
        ],
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Configured environment key or active hashed database key. Required scopes depend on the operation. Never put the key in a URL."
        }
    }
    
    for path_data in openapi_schema["paths"].values():
        for operation in path_data.values():
            if not isinstance(operation, dict) or "operationId" not in operation:
                continue
            # FastAPI emits this parameter from the actual authentication dependency.
            key_parameters = [parameter for parameter in operation.get("parameters", [])
                if parameter.get("in") == "header" and parameter.get("name", "").lower() == "x-api-key"]
            authenticated = bool(key_parameters)
            operation["security"] = [{"ApiKeyAuth": []}] if authenticated else []
            if authenticated:
                # ApiKeyAuth supplies this header in Swagger; a second parameter would duplicate it.
                operation["parameters"] = [parameter for parameter in operation.get("parameters", [])
                    if not (parameter.get("in") == "header" and parameter.get("name", "").lower() == "x-api-key")]
                operation["responses"].setdefault("401", {"description": "Invalid or inactive API key"})
                operation["responses"].setdefault("403", {"description": "Required scope or camera access denied"})
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "deepLinking": True,
        "displayRequestDuration": True,
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "tryItOutEnabled": True,
    },
)

# Override OpenAPI schema
app.openapi = custom_openapi

# CORS
if settings.CORS_ORIGINS:
    allow_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
elif settings.DEBUG:
    allow_origins = ["*"]  # dev convenience; restrict in production via CORS_ORIGINS
else:
    allow_origins = []  # secure default: no cross-origin access unless explicitly configured

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Location", "ETag", "Link", "Accept-Post", "Accept-Patch"],
)

# Artefact file serving (authenticated)
# Replaces the unauthenticated StaticFiles mount at /files.
@app.get("/files/{path:path}", response_class=StreamingResponse, responses={
    200: {"description": "Authorized artifact bytes; Content-Type depends on the stored file",
          "content": {media: {"schema": {"type": "string", "format": "binary"}} for media in
                      ("image/jpeg", "image/png", "application/json", "application/octet-stream")}},
    404: {"description": "Artifact missing or not registered for access"},
})
async def serve_artefact(
    path: str = FastAPIPath(..., description="Storage key path (e.g. cameras/123/outputs/abc.jpg)"),
    db: AsyncSession = Depends(get_db),
    api_key=Depends(verify_api_key),
):
    """Serve a stored artefact file with API-key authentication."""
    if "calibration-captures" in path.replace("\\", "/").split("/"):
        await calibration_captures.authorize_capture_artifact(db, path, api_key)
        return await calibration_captures.image_response(path)
    from api.services.prediction_access_service import authorize_artifact
    await authorize_artifact(db, path, api_key)
    storage = get_storage()
    if not storage.exists(path):
        raise HTTPException(status_code=404, detail="Artefact not found")
    file_stream = storage.get(path)
    if not file_stream:
        raise HTTPException(status_code=404, detail="Artefact not found in storage")
    # Guess content type from extension
    import mimetypes
    content_type, _ = mimetypes.guess_type(path)
    return StreamingResponse(
        file_stream,
        media_type=content_type or "application/octet-stream",
    )

# Include routers (all under /api/v1 prefix with proper sub-paths)
app.include_router(cameras.router, prefix="/api/v1/cameras")
app.include_router(streams.router, prefix="/api/v1/cameras/{camera_id}")
app.include_router(snapshots.router, prefix="/api/v1/cameras/{camera_id}/snapshots")
app.include_router(calibrations.router, prefix="/api/v1/cameras/{camera_id}/calibrations")
app.include_router(calibration_captures.router, prefix="/api/v1/cameras/{camera_id}/calibration-captures")
app.include_router(ground_refs.router, prefix="/api/v1/cameras/{camera_id}/ground-references")
app.include_router(ground_refs.model_router, prefix="/api/v1/cameras/{camera_id}/ground-models")
app.include_router(predictions.router, prefix="/api/v1/cameras/{camera_id}/predict")


@app.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint.

    Probes the database connection honestly instead of reporting connected
    unconditionally. Never exposes internal error details to the client.
    """
    try:
        await db.execute(text("SELECT 1"))
        database = "connected"
    except Exception:
        database = "disconnected"

    return {
        "status": "healthy" if database == "connected" else "degraded",
        "version": settings.APP_VERSION,
        "database": database,
    }


@app.get("/", response_model=RootResponse)
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG,
    )
