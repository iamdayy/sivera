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
- **Workers:** Celery/Redis under `api/workers/`, with separate `capture`, `calibration`, and `prediction` queues. PostgreSQL is the source of truth for progress and results.
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
       (grid medians, footprint hull × median height)
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
- `api/routes/streams.py`: stream desired state and authenticated WHEP proxy.
- `api/services/mediamtx_service.py`: bounded Control API/signaling requests to MediaMTX v1.20.0.
- `api/services/stream_service.py`: periodic desired-state reconciliation and expired-session cleanup.
- `api/routes/calibrations.py`: chessboard calibration and version activation.
- `api/routes/calibration_captures.py`: quality-selected RTSP capture, corner previews, and queued inactive calibration computation.
- `api/routes/snapshots.py`: original-resolution RTSP snapshots and freshness-bounded latest selection.
- `api/routes/ground_refs.py`: image/world correspondences, ground models, and BEV preview.
- `api/routes/predictions.py`: prediction submission, job/output retrieval, and result data.
- `api/services/`: domain and computer-vision logic.
- `api/services/prediction_worker_service.py`: durable prediction claims, source/freshness checks, completion fencing, and stale-job reconciliation.
- `api/workers/`: Celery tasks, per-task async database contexts, and killable OpenCV subprocesses.
- `api/alembic/`: schema migrations.

## 5. Domain invariants

- Use UUID identifiers wherever defined by current models and schemas.
- Calibration and ground-reference records are versioned per camera. Activation should not leave conflicting active versions for the same camera.
- Image points and world points must correspond one-to-one and meet the algorithm's minimum-point requirements.
- Prediction submission pins an active camera calibration, the latest ground model linked to an active reference for that calibration, and an active depth scale for that ground model. Versions from another camera/calibration/ground model must not be mixed. Activation changes do not rewrite already-pinned jobs.
- Keep image coordinates, world coordinates, camera pose, depth scale/bias, and physical units explicit. Never silently reinterpret coordinate systems.
- Distinguish volume estimation methods. The current grid method sums cell median heights × cell area; the method named `convex_hull` multiplies a 2D footprint-hull area by median height, not a true 3D hull volume.
- Missing or failed estimates must not be converted into zero.
- BEV is a top-down projection/visualization, not a complete true 3D reconstruction.
- Generate artifact paths and URLs through the storage abstraction rather than hardcoding filesystem paths in route handlers.

## 6. API contract and current caveats

- Authentication uses the `X-API-Key` header. Keys may be configured through `VALID_API_KEYS` and/or stored hashed in the database. Scope checks cover camera, calibration, ground-reference, and prediction operations.
- Prefer live `/openapi.json`, route declarations, and schema code over potentially stale README examples.
- Current route prefixes are mounted explicitly under `/api/v1/cameras` in `api/main.py`, including calibrations, ground references, streams, and predictions. Verify OpenAPI before changing them.
- Prediction supports `upload`, `rtsp_frame`, and `latest_rtsp` as multipart form sources. The route commits a pending job before Celery dispatch and returns HTTP 202; workers persist `processing -> done/failed`. Broker failure returns 503 and marks the pending job failed. Stale jobs fail rather than being automatically redispatched.
- RTSP sources disallow caller-supplied file/timestamp. They require enabled streams and original intrinsics with matching captured dimensions; legacy calibrations without original K require recalibration. Legacy upload predictions retain the historical already-undistorted input contract.
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
- Restrict CORS to known dashboard origins in production. Empty origins reject cross-origin access when DEBUG is false; DEBUG enables a wildcard fallback. Compose currently forces DEBUG=true for the API, so production must override it and set explicit origins.
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

## 11. MediaMTX integration (phase 1)

- `Camera.stream_enabled` persists desired state, default false. Source remains private; camera responses return `rtsp_url=null`, `rtsp_configured`, and `stream_enabled`. The dashboard must not overwrite an unchanged source with null.
- Exact source hostname/IP allowlist, RTSP/RTSPS only, TCP pulls and on-demand sources. Deployment must enforce camera-network egress limits independently of hostname validation.
- Owned paths are `sivera_<UUID>`. A PostgreSQL advisory transaction lock serializes reconcilers across API processes. Only canonical UUID namespace paths are removed, never unrelated paths. Failed sync is retried periodically; camera changes commit before background reconciliation.
- WHEP OPTIONS/POST/PATCH/DELETE require API key and camera:read, with camera-bound key restrictions. PostgreSQL `StreamSession` maps a public UUID to the private MediaMTX secret and an HMAC of the owning API key. PATCH/DELETE reauthenticate ownership, including across API workers/restarts. Absolute session TTL and cleanup are bounded by reconciliation; DELETE clears upstream before removing mapping.
- Upstream API and reader credentials are separate. No anonymous public read/publish. Loopback-only admin exists for container healthcheck. HTTP ports are localhost-only on the host; UDP ICE 8189 is published. Dashboard signaling is HTTPS through FastAPI in production; media goes directly to MediaMTX. TURN/TCP are deployment-specific.
- Forward only SDP, ETag, Accept-Post/Patch, ICE Link; rewrite Location to backend route. No upstream redirects/raw error details. POST is not retried. SQL parameters and HTTP WHEP secret URLs are redacted from normal logs.
- Playback requires a browser-compatible camera codec and reachable ICE candidate; signaling success alone does not verify playback. Capture and prediction use the original RTSP relay independently of browser playback.

## 12. RTSP calibration capture (phase 2)

- `CalibrationCaptureSession` and `CalibrationCaptureImage` persist UUIDs, validated chessboard parameters in meters, counters, source dimensions, UTC timestamps, deadline, heartbeat, safe error codes, and private storage keys. A PostgreSQL partial unique index permits one queued/running/stopping session per camera. Capture is independent of existing calibration records and never activates calibration.
- Authenticated JSON start (202), status, stop, paginated images, and JPEG download are mounted at `/api/v1/cameras/{camera_id}/calibration-captures`. All operations enforce calibration scopes and camera-bound key restrictions. Generic `/files` cannot bypass these restrictions for capture artifacts. No RTSP URLs, worker token, or S3 URLs enter capture responses.
- Legacy phase-2 sessions (`selection_version=0`) keep stored-frame targeting. New sessions use accepted-frame targeting (`selection_version=1`); `completed` still means capture target reached, not successful calibration. See phase 3 below.
- Session creation commits before queue dispatch. The persistent queued row acts as an outbox, redispatched at most every 15 seconds by the API startup/periodic reconciler. Duplicate delivery uses an atomic queued-to-running claim and per-run fencing token. Queue outage retains queued state until queue timeout or deadline; no reliance on Celery result backend for progress.
- Capture worker consumes only queue `capture`, concurrency 1 in Compose; calibration worker consumes `calibration`, concurrency 1; prediction worker consumes `prediction` and legacy `celery` to drain pre-deployment jobs. API and all workers share storage through `api/core/storage.py`. Async engines are created per task with NullPool to avoid cross-fork/cross-event-loop pooled connections. Capture transactions do not span FFmpeg or storage writes; prediction releases prerequisite reads before inference and commits outputs atomically on completion.
- FFmpeg pulls the authenticated original MediaMTX UUID relay over TCP, one JPEG per subprocess. No resize, crop, undistort, or autorotation. First-frame dimensions remain explicit; resolution/source changes fail the session. Timestamp records worker reception, not camera sensor time. Interval is the wait after each attempt. Frame bytes/pixels, attempts, subprocess duration, session deadline and Celery hard limit are bounded.
- Stop cancels the in-flight process group and retains committed images. Coreutils `timeout` imposes a wall-clock bound even if the Celery child dies; FFmpeg decoder also receives the configured pixel limit. Deadline includes queue time. Reconciliation reaps stale workers and fences subsequent writes; interrupted running sessions fail rather than silently restart. Counters allow interrupted attempts without inventing zero results.
- Local storage uses atomic replace; all services must use one writable root, with no silent fallback. New Compose volume requires operator backup/migration of pre-existing container-layer artifacts. S3 must remain private. Retention is not implemented; filesystem orphan cleanup and camera-deletion artifact cleanup are operator responsibilities, including crash between storage write and metadata commit.

## 13. Capture quality and calibration computation (phase 3)

- Migration `005_capture_quality_snapshots` adds quality counters, original-coordinate corners, pose descriptors, computation status/token/timestamps, explicit calibration intrinsics, snapshots, and persisted prediction options. Legacy rows retain `selection_version=0`, unassessed images, and null original K; migration does not fabricate original intrinsics from `k_matrix`.
- New captures store accepted and rejected frames, up to 300 per session. Chessboard axes require 3..30 inner corners, square length is meters, and resolution stays fixed. Rejection reasons include `chessboard_not_found`, `blur`, and `similar_pose`. Sharpness is Laplacian variance within the detected board; diversity compares normalized position, size, orientation, and perspective ratios against accepted views. Thresholds are heuristics, not proof of accurate calibration.
- `GET .../images/{image_id}/preview` draws detected corners on the original frame and requires `calibration:read`. `POST .../{session_id}/compute` requires `calibration:write` and a terminal capture with enough accepted frames (stored frames for legacy sessions, then quality assessed). Repeat requests do not duplicate queued/running/done computations; failed computations may be explicitly retried.
- `auto_calibrate=false` by default. True queues computation when capture completes, stops, or expires; insufficient usable views fail safely. Failed captures are not automatically computed. Computation has its own persistent outbox, atomic claim, token fence, queue timeout, and wall-clock timeout. Killable subprocesses with coreutils `timeout` bound OpenCV assessment/computation even after a Celery child dies.
- Camera-row locks serialize calibration version allocation and activation. Capture computations create inactive results and never replace active calibration. Existing multipart upload calibration remains synchronous and auto-activates on success; it now also stores `k_original`, `k_undist`, resolution, alpha, and per-view errors. Legacy `k_matrix` remains the undistorted K. RMS/per-view errors use pixels; stored coverage uses percentages.
- RTSP source HMAC is private, checked during capture/computation and again under a camera-row lock before final persistence. API and all workers require the same stable `SECRET_KEY`. Changing it invalidates old source identities and WHEP ownership. Relay configuration is checked before capture without modifying MediaMTX paths; unsynced source changes must reconcile first. Focus, zoom, camera pose, and firmware stream settings still need operator control; unchanged URLs cannot detect those physical changes.

## 14. RTSP snapshots and predictions (phase 4)

- `POST /api/v1/cameras/{camera_id}/snapshots` synchronously captures/persists an original-resolution JPEG (201, `camera:write`). `GET .../snapshots/latest` and `GET .../snapshots/{snapshot_id}/image` require `camera:read` and enforce camera-bound keys. Latest selection requires a matching source HMAC and `max_frame_age_sec` (default 30, range 1..300); explicit image download preserves access to older snapshots without a freshness check. Capture-session frames are not automatically snapshots.
- `rtsp_frame` captures a new frame when the prediction worker executes, not at submission. `latest_rtsp` checks snapshot availability/freshness at submission and selects the latest matching persisted snapshot again at execution. It does not capture a fallback frame or offer historical timestamps. Worker-reception UTC timestamps become `source_ts`; neither source claims sensor time.
- Original frames are never resized/cropped/undistorted during capture. The prediction pipeline validates exact calibration dimensions and finite original/target intrinsics, then undistorts into `k_undist`/legacy `k_matrix` before segmentation, ground mapping, and depth lifting. New explicit-calibration uploads use this same raw-image contract; legacy upload jobs without original intrinsics require already-undistorted input at registered camera dimensions. Ground-reference image points belong to the undistorted coordinate system.
- Jobs pin calibration/ground/depth IDs and persist segmentation, grid, freshness, and `return_outputs` options. Omitted `return_outputs` registers all 13 supported outputs; a list registers only selected types, and `[]` registers none. Internal processing files can still be generated. Job summaries, instances, and volumes are not filtered by this option. Output/download and generic `/files` authorization require camera-bound `predict:read` plus a registered output; capture/snapshot/calibration/ground previews keep their respective read scopes.
- Durable pending-to-processing claims ignore duplicate tasks. The worker rolls back failed processing writes, publishes only safe error codes, and rechecks source identity and processing status under row locks before completion. Reconciliation fails pending jobs after the queue timeout and processing jobs after the Celery hard limit plus 60 seconds. A dead worker is not resumed automatically; the operator submits a new job.
- Missing grid support or degenerate footprint hulls produce null summary/instance estimates and no corresponding `VolumeEstimate` row; numeric NPZ estimates use NaN. Grid allocation is capped at 1,000,000 cells. Volumes use m³/L, footprint m², and heights meters; neither method is a full 3D reconstruction.
- Disposable integration verifies real RTSP capture, migrations, storage, Celery transitions/failures, prerequisite compatibility, output selection, image encoding, geometry, and polling. Successful prediction integration mocks segmentation/depth inference; real camera accuracy, model weights/GPU inference, browser ICE/playback/reconnect, dashboard integration, TLS/TURN, and production deployment remain unverified.
