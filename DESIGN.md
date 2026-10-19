# SIVERA Backend — Design Specification

## 1. Purpose and ownership

SIVERA provides an API and processing pipeline for camera-based waste segmentation, monocular depth estimation, camera calibration, ground-plane mapping, Bird's Eye View (BEV) projection, and waste-volume estimation.

This repository (`iamdayy/sivera`) owns the backend API, persistence, storage integration, and computer-vision processing. The web dashboard is maintained separately in `iamdayy/sivera-dashboard`.

## 2. Architecture overview

- **API:** FastAPI application in `api/main.py`.
- **Configuration and security:** environment-driven settings and API-key validation under `api/core/`.
- **Persistence:** PostgreSQL with SQLAlchemy async sessions; spatial types use PostGIS/GeoAlchemy where configured.
- **Migrations:** Alembic under `api/alembic/`.
- **Storage:** storage abstraction in `api/core/storage.py`, intended to isolate local/S3-compatible artifact storage.
- **Computer vision:** OpenCV, NumPy/SciPy, segmentation, and monocular depth model integration.
- **Domain services:** calibration, ground model, depth, prediction orchestration, and volume estimation under `api/services/`.
- **Workers:** Celery/Redis-related components under `api/workers/`; verify actual route-to-worker wiring before assuming jobs are queued.
- **API documentation:** `/docs`, `/redoc`, and `/openapi.json`.
- **Health check:** `/health`.

Routers are included directly by `api/main.py`. Do not assume a global `/api/v1` prefix unless it is explicitly mounted.

## 3. Conceptual processing flow

```text
Camera + source image
        |
        v
Camera calibration (intrinsics/distortion)
        |
        v
Ground references (image points <-> world points)
        |
        v
Ground model (homography / pose / BEV parameters)
        |
        +--------------------------+
        |                          |
        v                          v
Instance segmentation       Monocular depth estimation
        |                          |
        +-------------+------------+
                      v
              3D lifting / mapping
                      |
                      v
               Volume estimation
             (grid/voxel, convex hull)
                      |
                      v
        Persist job, instances, estimates, artifacts
```

This is a conceptual view. Preserve the service interfaces and ordering actually implemented in source.

## 4. Important modules

- `api/main.py`: FastAPI lifecycle, middleware, router registration, static artifact serving, health/root endpoints, and OpenAPI customization.
- `api/core/config.py`: configuration and defaults.
- `api/core/database.py`: async engine/session lifecycle and database initialization.
- `api/core/security.py`: API-key validation and scope checks.
- `api/core/storage.py`: artifact storage abstraction.
- `api/models/orm.py`: SQLAlchemy entities, including cameras, calibrations, ground references/models, depth scales, prediction jobs/outputs, detected instances, volume estimates, and API keys.
- `api/schemas/`: request/response models.
- `api/routes/cameras.py`: camera registry and metadata.
- `api/routes/calibrations.py`: chessboard calibration and version activation.
- `api/routes/ground_refs.py`: image/world correspondences, ground models, and BEV preview.
- `api/routes/predictions.py`: prediction submission, job/output retrieval, and result data.
- `api/services/`: domain and computer-vision logic.
- `api/workers/`: background task components; confirm active usage.
- `api/alembic/`: schema migrations.

## 5. Domain invariants

- Use UUID identifiers wherever defined by current models and schemas.
- Calibration and ground-reference records are versioned per camera. Activation should not leave conflicting active versions for the same camera.
- Image points and world points must correspond one-to-one and meet the algorithm's minimum-point requirements.
- Prediction prerequisites include an active camera calibration, an active ground model/reference, and a depth scale.
- Keep image coordinates, world coordinates, camera pose, depth scale/bias, and physical units explicit. Never silently reinterpret coordinate systems.
- Distinguish volume estimation methods. Grid/voxel and convex-hull estimates can differ and must not be conflated.
- Missing or failed estimates must not be converted into zero.
- BEV is a top-down projection/visualization, not a complete true 3D reconstruction.
- Generate artifact paths and URLs through the storage abstraction rather than hardcoding filesystem paths in route handlers.

## 6. API contract and current caveats

- Authentication uses the `X-API-Key` header. Keys may be configured through `VALID_API_KEYS` and/or stored hashed in the database. Scope checks cover camera, calibration, ground-reference, and prediction operations.
- Prefer live `/openapi.json`, route declarations, and schema code over potentially stale README examples.
- Current route prefixes are mounted directly, including `/cameras`, `/cameras/{camera_id}/calibrations`, `/cameras/{camera_id}/ground-references`, and `/cameras/{camera_id}/predict`. Verify before changing them.
- Prediction accepts source types named `upload`, `rtsp_frame`, and `latest_rtsp`, but the current handler only processes uploads and raises `NotImplementedError` for RTSP sources.
- The prediction handler currently runs the pipeline synchronously despite returning HTTP 202 and containing a TODO about queueing work with Celery. Do not describe it as truly asynchronous until queue dispatch, status transitions, and polling are implemented and tested.
- Validate response models and ensure OpenAPI accurately reflects multipart fields and response shapes.
- Use stable, actionable HTTP errors. Do not expose stack traces, secrets, or internal filesystem paths in responses.

## 7. Errors and observability

- Use 4xx responses for invalid input, missing resources, unmet prerequisites, and authorization failures.
- Use 5xx responses for unexpected server failures; keep diagnostic details in server-side logs.
- Keep API errors predictable and machine-readable.
- Log request/job lifecycle, pipeline stage, duration, and failures with useful identifiers where possible.
- Never log API keys, RTSP credentials, or raw image contents.
- Health checks should accurately reflect dependency health; if semantics are expanded, check the database rather than reporting it connected unconditionally.

## 8. Security and deployment

- Store secrets in environment variables or a secret manager. Never commit real `.env` files, API keys, database credentials, or RTSP credentials.
- Restrict CORS to known dashboard origins in production. The current wildcard-origin configuration combined with credentials should be reviewed before deployment.
- Enforce upload size limits and validate actual file content, not just filename or MIME type.
- Avoid exposing arbitrary local filesystem paths through artifact endpoints.
- Use least-privilege API-key scopes and support key rotation/revocation.
- Keep model weights and generated artifacts out of source control unless deliberately required.
- Apply schema changes through reviewed migrations; do not rely on startup side effects to mutate production schema.
- Document container permissions, storage mounts, required environment settings, and external services.

## 9. Testing strategy

- Unit-test calibration, coordinate transforms, segmentation/depth adapters, volume methods, and schema validation.
- Test API authorization/scope enforcement, missing resources, prerequisite failures, upload validation, and output serialization.
- Test migrations against a disposable database.
- Use known geometry or synthetic points to verify metric units and volume calculations.
- Keep expensive inference out of ordinary unit tests; use mocks/fixtures for service logic and clearly marked integration tests for real model weights.
- Verify `/health` and `/openapi.json` in a configured test environment.

## 10. Definition of done

A backend change is complete when it preserves or documents API compatibility, includes appropriate tests and migrations, keeps units and coordinate assumptions explicit, handles errors and secrets safely, updates documentation when behavior changes, and reports which checks actually ran.
