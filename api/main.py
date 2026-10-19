"""FastAPI application factory and main entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

from api.core.config import settings
from api.core.database import init_db, close_db, engine
from api.core.storage import get_storage
from api.core.config import STORAGE_ROOT
from api.routes import cameras, calibrations, ground_refs, predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


# Custom OpenAPI schema with security
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "## Waste Volume Estimation API\n\n"
            "REST API for CCTV-based waste volume estimation using computer vision pipeline:\n"
            "- **Calibration**: Camera calibration from chessboard images\n"
            "- **Ground Reference**: Ground plane reference points for metric mapping\n"
            "- **Segmentation**: YOLO-seg waste detection (Sampah_Berserakan, Sampah_Overload)\n"
            "- **Depth Estimation**: Monocular depth estimation (YOLO depth)\n"
            "- **3D Lifting**: Lift waste pixels to 3D ground coordinates\n"
            "- **Volume Estimation**: Grid-based and convex hull volume calculation\n"
            "- **BEV Projection**: Bird's Eye View visualization\n\n"
            "### Authentication\n"
            "All endpoints require an API key via the `X-API-Key` header.\n"
            "Configure valid keys via `VALID_API_KEYS` environment variable.\n\n"
            "### Scopes\n"
            "- `camera:read` / `camera:write` - Camera management\n"
            "- `calibration:read` / `calibration:write` - Calibration\n"
            "- `ground_ref:read` / `ground_ref:write` - Ground references\n"
            "- `predict:read` / `predict:write` - Predictions\n\n"
            "### Versioning\n"
            "Calibrations and ground references are versioned per camera. "
            "Only one version can be active at a time."
        ),
        routes=app.routes,
        tags=[
            {"name": "cameras", "description": "Camera registry and management"},
            {"name": "calibrations", "description": "Camera calibration from chessboard images"},
            {"name": "ground-references", "description": "Ground plane reference points and homography"},
            {"name": "ground-models", "description": "Computed ground models and BEV preview"},
            {"name": "predictions", "description": "Waste detection, depth, volume estimation"},
        ],
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication. Set via VALID_API_KEYS env var."
        }
    }
    
    # Apply security to all endpoints by default
    for path_data in openapi_schema["paths"].values():
        for operation in path_data.values():
            if isinstance(operation, dict) and "operationId" in operation:
                operation["security"] = [{"ApiKeyAuth": []}]
    
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for local storage
app.mount("/files", StaticFiles(directory=STORAGE_ROOT), name="files")

# Include routers
app.include_router(cameras.router)
app.include_router(calibrations.router)
app.include_router(ground_refs.router)
app.include_router(ground_refs.model_router)
app.include_router(predictions.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "database": "connected",
    }


@app.get("/")
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