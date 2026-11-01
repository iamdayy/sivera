# SIVERA Backend: Design Specification

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
- **Runtime:** FastAPI lifespan owns scheduling and durable-job dispatch under `api/runtime/`. PostgreSQL domain rows are the durable queue; one bounded vision subprocess runs at a time. Celery, Redis, and Flower are removed.
- **API documentation:** `/docs`, `/redoc`, and `/openapi.json`.
- **Health check:** `/health`.

`api/main.py` explicitly mounts camera operations at `/api/v1/cameras`. `/files`, `/health`, and `/` are outside that prefix. `settings.API_PREFIX` does not control router mounting. The dashboard should configure the API origin once and keep exact operation paths in its HTTP layer.

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
- `api/models/orm.py`: SQLAlchemy entities, including cameras, calibrations, ground references/models, depth scales, registered previews, prediction jobs/outputs, detected instances, volume estimates, and API keys.
- `api/schemas/`: request/response models.
- `api/routes/cameras.py`: camera registry and metadata.
- `api/routes/streams.py`: stream desired state and authenticated WHEP proxy.
- `api/routes/public_stream.py`: public WHEP capabilities, offer/answer, ICE updates, and cleanup through the same MediaMTX service.
- `api/services/mediamtx_service.py`: bounded Control API/signaling requests to MediaMTX v1.20.0.
- `api/services/stream_service.py`: periodic desired-state reconciliation and expired-session cleanup.
- `api/routes/calibrations.py`: chessboard calibration and version activation.
- `api/routes/calibration_captures.py`: quality-selected RTSP capture, corner previews, and queued inactive calibration computation.
- `api/routes/snapshots.py`: original-resolution RTSP snapshots and freshness-bounded latest selection.
- `api/routes/ground_refs.py`: image/world correspondences, ground models, and BEV preview.
- `api/routes/predictions.py`: prediction submission, job/output retrieval, and result data.
- `api/services/`: domain and computer-vision logic.
- `api/services/prediction_runner_service.py`: durable prediction claims, source/freshness checks, completion fencing, and stale-job reconciliation.
- `api/runtime/`: FastAPI lifecycle runtime, event hub, queue admission, and manager for capture, calibration, prediction, and automatic jobs.
- `api/alembic/`: schema migrations.
- `api/scripts/export_openapi.py`, `openapi.json`: reproducible default-settings OpenAPI export for contract review/dashboard integration.

## 5. Domain invariants

- Use UUID identifiers wherever defined by current models and schemas.
- Calibration and ground-reference records are versioned per camera. Activation should not leave conflicting active versions for the same camera.
- Image points and world points must correspond one-to-one and meet the algorithm's minimum-point requirements.
- Prediction submission pins an active camera calibration, the latest ground model linked to an active reference for that calibration, and an active depth scale for that ground model. Versions from another camera/calibration/ground model must not be mixed. Activation changes do not rewrite already-pinned jobs.
- Public depth-scale provisioning is available under the selected ground model. Upload a test image, inspect its undistorted preview, and submit camera-axis Z measurements. Creation auto-activates a fit that passes numerical/residual gates; this is not certification of physical accuracy. Camera/calibration/ground setup alone still does not prove prediction readiness. Ground-model `/active` now uses the same compatible calibration/reference selection as prediction.
- Keep image coordinates, world coordinates, camera pose, depth scale/bias, and physical units explicit. Never silently reinterpret coordinate systems.
- Distinguish volume estimation methods. The current grid method sums cell median heights × cell area; the method named `convex_hull` multiplies a 2D footprint-hull area by median height, not a true 3D hull volume.
- Missing or failed estimates must not be converted into zero.
- BEV is a top-down projection/visualization, not a complete true 3D reconstruction.
- Generate artifact paths and URLs through the storage abstraction rather than hardcoding filesystem paths in route handlers.

## 6. API contract and current caveats

- Authentication uses the `X-API-Key` header. Keys may be configured through `VALID_API_KEYS` and/or stored hashed in the database. Scope checks cover camera, calibration, ground-reference, and prediction operations.
- Public endpoints include `/`, `/health`, `/docs`, `/redoc`, `/openapi.json`, and camera playback under `/api/v1/public/cameras/{camera_id}/stream/whep`. `/files` and operations under `/api/v1/cameras` require authentication. Protected operations return 422 for missing headers, 401 for invalid/inactive/expired database keys, and 403 for scope/camera denial. Environment-configured keys have wildcard scopes and no database expiry; they are not appropriate browser-build secrets.
- Prefer live `/openapi.json`, route declarations, and schema code over potentially stale README examples.
- Current route prefixes are mounted explicitly under `/api/v1/cameras` in `api/main.py`, including calibrations, ground references, streams, and predictions. Verify OpenAPI before changing them.
- Prediction supports `upload`, `rtsp_frame`, and `latest_rtsp` as multipart form sources. The route commits a pending job before runtime dispatch and returns HTTP 202; the runtime runner persists `processing -> done/failed`. Queue-full returns 503 and marks the pending job failed. Stale jobs fail rather than being automatically redispatched.
- RTSP sources disallow caller-supplied file/timestamp. They require enabled streams and original intrinsics with matching captured dimensions; legacy calibrations without original K require recalibration. Legacy upload predictions retain the historical already-undistorted input contract.
- Validate response models and ensure OpenAPI accurately reflects multipart fields and response shapes.
- `estimated_duration_sec=30` is a fixed advisory submission field, not a measured duration, progress percentage, or SLA. Persistent job listing is available per camera with `page`, `page_size`, and optional `status`; deletion/cancellation endpoints are not available.
- Use stable, actionable HTTP errors. Do not expose stack traces, secrets, or internal filesystem paths in responses.

## 7. Errors and observability

- Use 4xx responses for invalid input, missing resources, unmet prerequisites, and authorization failures.
- Use 5xx responses for unexpected server failures; keep diagnostic details in server-side logs.
- Keep API errors predictable and machine-readable.
- Log request/job lifecycle, pipeline stage, duration, and failures with useful identifiers where possible.
- Never log API keys, RTSP credentials, or raw image contents.
- `/health` probes PostgreSQL and returns HTTP 200 with `healthy/connected` or `degraded/disconnected`; clients must read the body, not only the status code. It does not test API-key validity, Redis, workers, MediaMTX, model loading, or camera availability.
- Error `detail` is not uniform: legacy strings, safe `{code: ...}` objects, and FastAPI validation lists coexist. Clients must normalize all three without discarding machine-readable codes; no schema migration or global error-envelope change is part of this integration.

## 8. Security and deployment

- Store secrets in environment variables or a secret manager. Never commit real `.env` files, API keys, database credentials, or RTSP credentials.
- Restrict CORS to known dashboard origins in production. Empty origins reject cross-origin access when DEBUG is false; DEBUG enables a wildcard fallback. Compose currently forces DEBUG=true for the API, so production must override it and set explicit origins.
- Enforce upload size limits and validate actual file content, not just filename or MIME type.
- Avoid exposing arbitrary local filesystem paths through artifact endpoints.
- Use least-privilege API-key scopes and support key rotation/revocation.
- Keep model weights and generated artifacts out of source control unless deliberately required.
- Apply schema changes through reviewed migrations; do not rely on startup side effects to mutate production schema.
- Document container permissions, storage mounts, required environment settings, and external services.
- API-key hashes, WHEP ownership, and source HMAC depend on the same stable `SECRET_KEY`; replacing it without a migration invalidates stored key hashes and identities.
- Ground-reference/model/depth-scale routers enforce camera-bound keys. Database authentication checks UTC expiry. Generic artifact access requires a registered resource and domain read scope; unknown namespaces and orphan keys are denied. Multi-organization ownership/isolation remains outside the current product scope.

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
- Per the user's 26 October 2026 decision, all RTSP/RTSPS hostnames/IPs are accepted and `MEDIAMTX_ALLOWED_RTSP_HOSTS` is unused. URL scheme/host/port/fragment validation, TCP pulls, on-demand sources, authentication, scopes, and camera binding still apply.
- Owned paths are `sivera_<UUID>`. A PostgreSQL advisory transaction lock serializes reconcilers across API processes. Only canonical UUID namespace paths are removed, never unrelated paths. Failed sync is retried periodically; camera changes commit before background reconciliation.
- WHEP under `/api/v1/cameras/{camera_id}` requires an API key and camera:read, with camera-bound key restrictions. PostgreSQL `StreamSession` maps a public UUID to the private MediaMTX secret and an HMAC of the owning API key. PATCH/DELETE reauthenticate ownership, including across API restarts. Absolute session TTL and cleanup are bounded by reconciliation; DELETE clears upstream before removing mapping.
- Upstream API and reader credentials are separate. No anonymous public read/publish. Loopback-only admin exists for container healthcheck. HTTP ports are localhost-only on the host; UDP ICE 8189 is published. Dashboard signaling is HTTPS through FastAPI in production; media goes directly to MediaMTX. TURN/TCP are deployment-specific.
- Forward only SDP, ETag, Accept-Post/Patch, ICE Link; rewrite Location to backend route. No upstream redirects/raw error details. POST is not retried. SQL parameters and HTTP WHEP secret URLs are redacted from normal logs.
- Playback requires a browser-compatible camera codec and reachable ICE candidate; signaling success alone does not verify playback. Capture and prediction use the original RTSP relay independently of browser playback.

## 12. RTSP calibration capture (phase 2)

- `CalibrationCaptureSession` and `CalibrationCaptureImage` persist UUIDs, validated chessboard parameters in meters, counters, source dimensions, UTC timestamps, deadline, heartbeat, safe error codes, and private storage keys. A PostgreSQL partial unique index permits one queued/running/stopping session per camera. Capture is independent of existing calibration records and never activates calibration.
- Authenticated JSON start (202), status, stop, paginated images, and JPEG download are mounted at `/api/v1/cameras/{camera_id}/calibration-captures`. All operations enforce calibration scopes and camera-bound key restrictions. Generic `/files` cannot bypass these restrictions for capture artifacts. No RTSP URLs, worker token, or S3 URLs enter capture responses.
- Legacy phase-2 sessions (`selection_version=0`) keep stored-frame targeting. New sessions use accepted-frame targeting (`selection_version=1`); `completed` still means capture target reached, not successful calibration. See phase 3 below.
- Session creation commits before queue dispatch. The persistent queued row acts as an outbox, redispatched at most every 15 seconds by the API runtime loop. Duplicate delivery uses an atomic queued-to-running claim and per-run fencing token. Queue outage retains queued state until queue timeout or deadline.
- The runtime loop runs capture (concurrency 1), calibration compute (concurrency 1), prediction (concurrency 1), and automatic inference (one vision computation at a time) fairly from PostgreSQL domain rows. API and all jobs share storage through `api/core/storage.py`. Capture transactions do not span FFmpeg or storage writes; prediction releases prerequisite reads before inference and commits outputs atomically on completion.
- FFmpeg pulls the authenticated original MediaMTX UUID relay over TCP, one JPEG per subprocess. No resize, crop, undistort, or autorotation. First-frame dimensions remain explicit; resolution/source changes fail the session. Timestamp records runtime reception, not camera sensor time. Interval is the wait after each attempt. Frame bytes/pixels, attempts, subprocess duration, session deadline are bounded by coreutils `timeout` and the vision subprocess timer.
- Stop cancels the in-flight process group and retains committed images. Coreutils `timeout` imposes a wall-clock bound even if the subprocess dies; FFmpeg decoder also receives the configured pixel limit. Deadline includes queue time. Reconciliation reaps stale jobs and fences subsequent writes; interrupted running jobs fail rather than silently restart. Counters allow interrupted attempts without inventing zero results.
- Local storage uses atomic replace; all services must use one writable root, with no silent fallback. New Compose volume requires operator backup/migration of pre-existing container-layer artifacts. S3 must remain private. Retention is not implemented; filesystem orphan cleanup and camera-deletion artifact cleanup are operator responsibilities, including crash between storage write and metadata commit.

## 13. Capture quality and calibration computation (phase 3)

- Migration `005_capture_quality_snapshots` adds quality counters, original-coordinate corners, pose descriptors, computation status/token/timestamps, explicit calibration intrinsics, snapshots, and persisted prediction options. Legacy rows retain `selection_version=0`, unassessed images, and null original K; migration does not fabricate original intrinsics from `k_matrix`.
- New captures store accepted and rejected frames, up to 300 per session. Chessboard axes require 3..30 inner corners, square length is meters, and resolution stays fixed. Rejection reasons include `chessboard_not_found`, `blur`, and `similar_pose`. Sharpness is Laplacian variance within the detected board; diversity compares normalized position, size, orientation, and perspective ratios against accepted views. Thresholds are heuristics, not proof of accurate calibration.
- `GET .../images/{image_id}/preview` draws detected corners on the original frame and requires `calibration:read`. `POST .../{session_id}/compute` requires `calibration:write` and a terminal capture with enough accepted frames (stored frames for legacy sessions, then quality assessed). Repeat requests do not duplicate queued/running/done computations; failed computations may be explicitly retried.
- `auto_calibrate=false` by default. True queues computation when capture completes, stops, or expires; insufficient usable views fail safely. Failed captures are not automatically computed. Computation has its own persistent outbox, atomic claim, token fence, queue timeout, and wall-clock timeout. Killable subprocesses with coreutils `timeout` bound OpenCV assessment/computation even after an abrupt API process death.
- Camera-row locks serialize calibration version allocation and activation. Capture computations create inactive results and never replace active calibration. Existing multipart upload calibration remains synchronous and auto-activates on success; it now also stores `k_original`, `k_undist`, resolution, alpha, and per-view errors. Legacy `k_matrix` remains the undistorted K. RMS/per-view errors use pixels; stored coverage uses percentages.
- Calibration preview repair (29 October 2026): the on-demand undistort endpoint previously returned generated/cached files without `PreviewArtifact` registration, so authenticated `/files` returned 404 even when bytes existed. It now locks the calibration row, reuses only registered existing previews, regenerates unregistered files from the source, and commits registration before returning a URL. A registered preview with missing bytes is regenerated without duplicate metadata. API paths/schemas and calibration activation remain the same.
- Preview verification: 42 focused tests passed (`test_calibration_preview.py`, `test_prediction_access.py`, `test_api.py`, with mocked database dependencies), and two `test_calibration_preview_integration.py` cases passed against disposable PostGIS with real migrations, database API keys, local storage, OpenCV, and ASGI HTTP. They cover concurrent generation, legacy cache recovery, missing-file recovery, repeat downloads, scope/camera denial, and parent deletion. Live local backend returned HTTP 200 for metadata and JPEG; the dashboard's Chromium preview action opened and decoded the real 1920×1080 image without page errors. S3 and physical calibration accuracy were not tested in this repair.
- RTSP source HMAC is private, checked during capture/computation and again under a camera-row lock before final persistence. API and all runtime jobs require the same stable `SECRET_KEY`. Changing it invalidates old source identities and WHEP ownership. Relay configuration is checked before capture without modifying MediaMTX paths; unsynced source changes must reconcile first. Focus, zoom, camera pose, and firmware stream settings still need operator control; unchanged URLs cannot detect those physical changes.

## 14. RTSP snapshots and predictions (phase 4)

- `POST /api/v1/cameras/{camera_id}/snapshots` synchronously captures/persists an original-resolution JPEG (201, `camera:write`). `GET .../snapshots/latest` and `GET .../snapshots/{snapshot_id}/image` require `camera:read` and enforce camera-bound keys. Latest selection requires a matching source HMAC and `max_frame_age_sec` (default 30, range 1..300); explicit image download preserves access to older snapshots without a freshness check. Capture-session frames are not automatically snapshots.
- `rtsp_frame` captures a new frame when the runtime runner executes, not at submission. `latest_rtsp` checks snapshot availability/freshness at submission and selects the latest matching persisted snapshot again at execution. It does not capture a fallback frame or offer historical timestamps. Runtime-reception UTC timestamps become `source_ts`; neither source claims sensor time.
- Original frames are never resized/cropped/undistorted during capture. The prediction pipeline validates exact calibration dimensions and finite original/target intrinsics, then undistorts into `k_undist`/legacy `k_matrix` before segmentation, ground mapping, and depth lifting. New explicit-calibration uploads use this same raw-image contract; legacy upload jobs without original intrinsics require already-undistorted input at registered camera dimensions. Ground-reference image points belong to the undistorted coordinate system.
- Jobs pin calibration/ground/depth IDs and persist segmentation, grid, freshness, and `return_outputs` options. Omitted `return_outputs` registers all 13 supported outputs; a list registers only selected types, and `[]` registers none. Internal processing files can still be generated. Job summaries, instances, and volumes are not filtered by this option. Output/download and generic `/files` authorization require camera-bound `predict:read` plus a registered output; capture/snapshot/calibration/ground previews keep their respective read scopes.
- Durable pending-to-processing claims ignore duplicate tasks. The runtime runner rolls back failed processing writes, publishes only safe error codes, and rechecks source identity and processing status under row locks before completion. Reconciliation fails pending jobs after the queue timeout and processing jobs after the prediction timeout plus grace seconds. A dead runtime is not resumed automatically; the operator submits a new job.
- Missing grid support or degenerate footprint hulls produce null summary/instance estimates and no corresponding `VolumeEstimate` row; numeric NPZ estimates use NaN. Grid allocation is capped at 1,000,000 cells. Volumes use m³/L, footprint m², and heights meters; neither method is a full 3D reconstruction.
- Disposable integration verifies real RTSP capture, migrations, storage, runtime job transitions/failures, prerequisite compatibility, output selection, image encoding, geometry, and polling. Successful prediction integration mocks segmentation/depth inference; real camera accuracy, model weights/GPU inference, browser ICE/playback/reconnect, dashboard integration, TLS/TURN, and production deployment remain unverified.

## 15. Published API contract and dashboard handoff

| Domain | Paths relative to `/api/v1/cameras/{camera_id}` | Submission/completion contract |
|---|---|---|
| Stream | `GET/PATCH /stream`, `OPTIONS/POST /stream/whep`, `PATCH/DELETE /stream/whep/{session_id}` | PATCH stores desired state; GET reports synchronization/runtime separately. Offer POST returns 201 SDP plus backend Location; PATCH returns 204 or 200 ICE-restart fragment; DELETE returns 204. |
| Calibration capture | `POST /calibration-captures`, `GET /calibration-captures/{session_id}`, `POST /calibration-captures/{session_id}/stop`, `GET /calibration-captures/{session_id}/images`, image/preview GET | JSON start returns 202 and `session_id`; status uses accepted/rejected counters, deadline, and independent computation state. Image pagination uses `offset`/`limit`. There is no session-list endpoint. |
| Capture computation | `POST /calibration-captures/{session_id}/compute` | 202; poll `compute_status` on the capture status endpoint. Terminal capture and enough selected views required. Fetch resulting calibration, inspect, then explicitly activate. |
| Snapshot | `POST /snapshots`, `GET /snapshots/latest`, `GET /snapshots/{snapshot_id}/image` | Synchronous 201 capture; latest GET never captures. Default age 30 seconds, range 1..300. Authenticated JPEG download is separate from metadata. |
| Prediction | `POST/GET /predict`, job GET and `/outputs`, `/outputs/{output_type}`, `/instances`, `/volumes` | Multipart POST returns 202 and `job_id`; GET job reports `pending/processing/done/failed`. Output selection affects registered downloads, not result summaries. |
| Depth scale | `POST/GET /ground-models/{ground_model_id}/depth-scales`, `GET .../active`, `GET /{depth_scale_id}`, `POST /{depth_scale_id}/activate`; `POST /test-images`, `GET /test-images/{image_id}/image` | Multipart image upload returns 201 and an authenticated undistorted PNG preview. JSON creation runs bounded CPU depth inference, returns 201, and auto-activates only accepted fits. Activation returns 200. Active calibration/reference compatibility is rechecked before writes. |

All camera/job/session/image/snapshot IDs are UUIDs. Capture uses `session_id` in responses; prediction submission uses `job_id`, while persisted job reads use `id`. Stream desired `enabled`, runtime `status`, and `synced` are distinct. `idle` is not proof the camera is online or offline; `online` is not proof a browser has decoded video.

Preview/download requests require header authentication; an `<img src>` or ordinary link cannot attach `X-API-Key`. Dashboard adapters should fetch bytes into revocable blob URLs or use a deliberately authenticated server-side proxy. Prefer dedicated capture/snapshot/output URLs to raw storage keys. Preserve backend `Location`/`ETag` and the owning key across WHEP cleanup. Never send application keys to untrusted URL origins.

OpenAPI custom generation derives security from FastAPI's actual `X-API-Key` dependency parameter, removes that duplicate header parameter after registering `ApiKeyAuth`, and leaves public operations with `security: []`. Root/health now have explicit response schemas; binary artifact/output endpoints expose their actual content types; WHEP documents SDP, ICE fragments, response headers, and optional `If-Match`. Field documentation preserves raw/undistorted coordinates, capture/compute distinctions, freshness, and multipart output selection without changing their wire formats.

The original contract-only handoff added no persistence fields. The 26 October alignment adds revisions 006 and 007, described below; current migration head is `007_preview_artifacts`.

Generate the contract from the repository root:

```bash
python api/scripts/export_openapi.py openapi.json
```

The exporter imports the app in a child process with an empty environment and a temporary working directory without `.env`, without contacting PostgreSQL/storage/MediaMTX or starting lifespan tasks. This also bypasses invalid local settings before the initial configuration import. The checked-in snapshot is reproducible and secret-free; a deployment can change configuration-derived defaults or dependency-version serialization, so live `/openapi.json` remains authoritative. The backend publishes `openapi.json`; dashboard adapters/types live in `../sivera-dashboard`. The 26 October alignment updates that separate repository's source-key and identity-recovery handling. See `docs/depth-scale-dashboard-handoff.md`. Historical plans may describe obsolete prefixes or RTSP behavior.

## 16. Verification and local database operation (22 October 2026)

Implementation commits are `704c7d3` (phases 1–2) and `b775270` (phases 3–4); they have not been pushed. Final re-verification after the contract/documentation update ran **242 passing local tests with 12 opt-in integration skips**, **9 passing capture/calibration/prediction integration tests**, and **3 passing stream/WHEP integration tests**. Integration resources were disposable PostgreSQL/PostGIS, MediaMTX `bluenviron/mediamtx:1.20.0-ffmpeg`, synthetic H264 RTSP, storage, and the FastAPI runtime. Prediction inference was mocked; camera capture, migrations, encoding, geometry, and persisted polling were real. The live ASGI OpenAPI paths/components matched the published contract after excluding configuration-dependent defaults. Export reproducibility, resistance to invalid environment/`.env` settings, local references/unique operation IDs, component JSON Schema syntax, multipart fields, WHEP headers, compile, migration head, Compose structure with temporary credentials, and diff whitespace were checked. Starlette TestClient and Compose's obsolete version field produce existing deprecation warnings. Detailed commands/results are recorded in the local backend plan; frontend build/browser tests are not claimed.

The user separately authorized discarding SIVERA application data in `waste_db` belonging to `waste-postgres`. Because local PostgreSQL occupied port 5432, a network-isolated maintenance container used the original image ID and `sivera_pgdata` volume while the original container stayed stopped. Ten legacy application tables had no Alembic history. They were dropped in dependency order with RESTRICT, then migrations 001 through `005_capture_quality_snapshots` ran in one transaction, without a guessed stamp or broad schema drop.

Final database verification found the expected revision and 14 empty application tables, including database API keys; ORM column types/nullability and principal constraints/active-capture uniqueness matched. PostGIS extensions, 37 reference tables, and 8,500 `spatial_ref_sys` rows were retained. Artifact files, other databases, `.env`, local PostgreSQL, and the original container configuration were not changed. The maintenance container was stopped/removed without deleting the volume. This is a dated operation record, not a destructive setup recipe or proof of the current deployed database state.

At that inspection, `waste-postgres` remained stopped with its 5432 binding conflict unresolved; local `DATABASE_URL` reached the other PostgreSQL and rejected authentication. Database API keys needed provisioning again, and retained artifact files could be orphaned. MediaMTX passwords were empty in the inspected `.env`; deployment needed credentials, shared non-default secret, correct database target/port, writable shared storage, CORS/HTTPS/ICE settings, model weights, and real camera/browser verification. This is the 22 October record, not a fresh deployment probe.

## 17. Depth-scale provisioning repair (24 October 2026)

Historical implementation record; the 26 October alignment below supersedes its identity and preview-persistence details.

- `/files` is download-only. Test-image upload accepts JPEG/PNG/BMP, bounded by upload bytes and decoded pixels, at exact calibration dimensions. Explicit calibrations receive raw images and undistort before both preview and model inference. Legacy calibrations require already-undistorted images at camera dimensions. Incomplete original intrinsics, invalid matrices, and target K inconsistent with registered K are rejected.
- Upload returns a ground-model/camera-bound source key and authenticated PNG preview. Original bytes and preview use the shared storage backend, including private S3. JSON creation accepts these source keys or exact keys registered in the selected calibration's `calib_images`; it rejects filesystem paths, URLs, arbitrary keys, and cross-camera images. BEV uses storage keys and the same bounded image preparation; optional masks are already-undistorted at matching dimensions and require their own artifact read scope.
- Measurements are 3..100 distinct integer undistorted `(x,y)` pixels with finite positive camera-axis Z in meters, within ordered clipping bounds. They are not Euclidean distance along the pixel ray. No point is silently dropped. Fits reject nonfinite/nonpositive model depth, out-of-bounds points, effectively constant raw/reference depths, rank deficiency, nonpositive scale, and poor residuals.
- Default residual gates: mean absolute 0.25 m, maximum absolute 0.5 m, maximum per-sample relative 0.1. Settings expose these heuristics. Residuals are signed `fitted - measured`; aggregates are absolute. These are in-sample diagnostics, not held-out accuracy. This additional linear correction is not Ultralytics `model.calibrate()` and does not establish camera-specific metric accuracy.
- Installed `.pt` models resolve strictly within `MODELS_DIR`, including symlink confinement. Creation uses the real synchronous `predict_depth()` adapter in a killable subprocess, CPU and prediction's image size. Wall-clock default is 120 seconds; one creation inference per API process runs at a time, others return 503 `depth_scale_busy`. There is no depth-scale queue/job/polling contract and no automatic retry or idempotency guarantee.
- Creation releases database reads before inference, then locks camera followed by ground model and rechecks active compatibility. Creation and activation deactivate competing scales atomically under those locks. Activation rejects invalid saved fit diagnostics and unavailable model files. Historical duplicate active rows are read deterministically; public writes normalize the selected ground model without a destructive data migration. Pinned prediction jobs remain unchanged.
- No ORM fields or migration are added; head remains `005_capture_quality_snapshots`. Images have a backend-controlled UUID namespace, not a new table. Access checks require the parent camera/ground model and ground scope, including generic `/files`. Files remain after parent deletion or partial upload; retention/physical cleanup remains an operator task.
- Docker dependencies now explicitly install depth-capable Ultralytics 8.4.182, Torch 2.5.0+cpu/torchvision 0.20.0+cpu, aligned OpenCV wheels, and SciPy used by volume geometry. Model weights, deployment restarts, user databases, MediaMTX image/config, and dashboard source are not changed by this repair.

## 18. Measurement correctness and validation requirements

This section describes required verification, not a claim that every item is already implemented or physically validated.

### Depth output and 3D lifting

- Record the selected model artifact/version and its documented output convention. Metric units alone do not tell whether a pixel value represents camera-axis Z-depth, radial distance, or relative depth.
- The current pinhole lifting form `X=(u-cx)D/fx`, `Y=(v-cy)D/fy`, `Z=D` is valid only for camera-axis Z-depth and matching undistorted image coordinates/intrinsics. If the model returns radial range, convert using the normalized camera ray; if relative depth, do not describe the output as metric without a validated metric scale.
- Keep image preparation identical between scale fitting and prediction (including undistortion, image dimensions, channel order, and model preprocessing).
- Bind every depth-scale fit to the exact depth model identity/version and ground model. Confirm at inference time that the same model is selected. Replacing weights or changing the output convention must invalidate or explicitly revalidate the scale.
- Treat fitted scale/bias and residual thresholds as calibration diagnostics, not independent proof of physical accuracy. Evaluate on held-out measurements, not only the samples used to fit the transform.

### Segmentation output contract

- The segmentation adapter must explicitly support the configured model's real output type: instance masks versus semantic label map. Validate the adapter with a known image and known class IDs/names.
- Define whether waste classes are mutually exclusive physical regions or scene-condition labels. If masks overlap or describe the same pile at different levels, deduplicate/union regions before volume aggregation to avoid double counting.
- Validate morphology and component-size thresholds against small, thin, low-contrast, and partially occluded waste. Keep thresholds configurable only where justified and test their effect on retained area.

### Volume methods

- **Grid method:** define cell size, per-cell height statistic, minimum sample count, empty-cell/hole policy, and boundary-cell area policy. Test several resolutions (suggested starting points: 0.025 m, 0.05 m, and 0.10 m) and density thresholds. Quantify error against physical reference volumes.
- **Footprint × height proxy:** if the implementation uses a 2D convex hull area multiplied by a median/representative height, name it as a footprint-height approximation. It is not a full 3D convex hull volume and should not be described as such.
- Compare candidate methods using identical scenes and a defined physical ground-truth procedure. Report absolute error in `m³` and relative error in percent. Choose the primary display method only after validation on representative pile shapes and sizes.
- Handle empty segmentation, sparse points, holes, degenerate hulls, non-finite depths, and clipped depths explicitly. Missing estimates must remain null/unavailable rather than becoming zero.
- Bound memory and execution time at camera resolution; use spatial binning/vectorization where it preserves the method's defined semantics.

### Job lifecycle and API-to-UI contract

- Prediction state is persisted as `pending -> processing -> done/failed`. Completion must be fenced so a stale/late worker cannot overwrite a terminal state or publish partial outputs as complete.
- Distinguish HTTP acceptance (202) from completed inference. The UI must poll the persisted job status; a fixed `estimated_duration_sec` is advisory, not actual progress or a service-level guarantee.
- Keep error codes stable and actionable. The dashboard should show safe human-readable recovery guidance without exposing internal paths, stack traces, or credentials.
- Keep the OpenAPI contract, backend schemas/routes/tests, and dashboard API adapters synchronized when changing readiness, depth-scale, prediction status, or output metadata.

### Validation dataset and release gate

Before claiming that volume estimation is accurate:
1. Collect representative waste piles spanning different sizes, shapes, materials, camera distances, lighting, and occlusion.
2. Measure reference volume using a documented, repeatable physical method.
3. Split scenes into calibration/tuning and held-out evaluation sets; do not tune and report accuracy on the same samples.
4. Record grid cell size, sparse-cell threshold, segmentation configuration, depth model/version, scale fit, and camera calibration for each run.
5. Report MAE/RMSE and relative error where appropriate, alongside failures and excluded samples.
6. Document limits and repeatability. A successful API job or accepted depth-scale fit is not a substitute for this evaluation.


## 19. Measurement and contract alignment (26 October 2026)

- Creation now accepts `test_image`, `depth_model`, `clip_min_m`, and `clip_max_m` alongside measurements. An uploaded source key selects that image even if the calibration has another image or none. Omitting it retains the first-calibration-image fallback. Source authorization and exact raw/undistorted dimensions apply before inference.
- Migration `006_depth_model_identity` adds nullable JSON identity and fit provenance. Identity contains weights SHA256, Ultralytics/Torch/OpenCV versions, preprocessing version, camera-axis convention, and inference size. Provenance stores the source key, prepared-image SHA256, measured points, and acceptance thresholds privately. Existing rows remain null and require a new measured fit.
- Creation, active GET, activation, prediction submission, and depth execution check the saved identity against installed weights/preparation. The depth cache reloads a changed file at the same path; changes during inference are rejected. Active GET/activation return 409 for missing identity, changed model, or invalid saved fit and 503 for unavailable weights. Historical list/detail reads remain available.
- [Ultralytics depth documentation](https://docs.ultralytics.com/tasks/depth/) explicitly distinguishes viewing-axis depth from rangefinder distance and describes a global calibration baked into released YOLO26 depth weights. The installed 8.4.182 `DepthPredictor` stretches inputs and returns an image-aligned map. SIVERA applies its existing additional linear correction; whether that improves held-out physical measurements is still unverified. A fingerprint records model identity, not the truth of custom-model training labels or physical accuracy.
- Migration `007_preview_artifacts` registers uploaded depth source/preview pairs, calibration previews, and BEV previews against exactly one calibration or ground model. Foreign keys cascade metadata on parent deletion. Original calibration images remain registered through `calib_images`. No legacy preview ownership is guessed; regenerate old unregistered previews. Files survive metadata deletion and interrupted writes, and physical cleanup stays an operator task.
- Database-key expiry uses UTC and rejects equality with the expiry instant. `/files` has no permissive unknown-namespace fallback; resource registration, camera binding, and domain scope are required. Upload/RTSP decoders select accepted Pillow formats directly, avoiding Ultralytics' global `Image.open` plugin auto-install path on corrupt input.
- Instance segmentation validates task, mask/box consistency, and class IDs. Aggregate volume uses the union of selected waste masks. Mask metadata records union/overlapping-membership pixels and morphology additions/removals. Per-instance results can overlap; operational class definitions and real-image morphology performance remain unverified.
- Grid grouping sorts cell IDs instead of rescanning every point per cell. Cell median, minimum support (default two), holes (NaN, no interpolation), full boundary-cell area, and the 1,000,000-cell cap are preserved. Unsupported/nonpositive aggregate estimates remain unavailable. Synthetic prisms cover 0.025/0.05/0.10 m cells and multiple support thresholds. A local 100,000-point 80×80-grid benchmark (median of three runs) measured 0.510247 s before and 0.088148 s after, with identical grids; this is not a production throughput result.

### Docker verification (26 October 2026)

- Current source mounted read-only into the existing backend image passed **354 tests, with 15 opt-in skips**. A separate disposable PostgreSQL/PostGIS and MediaMTX run passed **14 integration tests, with 1 skip** requiring real depth weights. PostgreSQL was 16.4 with PostGIS 3.4.3; MediaMTX remained `1.20.0-ffmpeg`.
- Real-database checks covered migration upgrade/downgrade through `007_preview_artifacts`, legacy-null identity, JSONB artifact authorization, preview parent constraints/cascades, concurrent depth activation, and prediction pins. Synthetic RTSP capture, WHEP signaling, queue/failure paths, and a disposable worker-child crash were exercised.
- HTTP smoke with application lifespan enabled returned healthy/connected `/health` and an exact OpenAPI snapshot match. Database-key expiry, missing/invalid credentials, scope/camera denials, registered/orphan file access, legacy identity recovery, upload, authenticated preview, and metadata cascade behavior passed.
- Successful prediction integration uses mocked depth/segmentation inference. Real weights, browser ICE/playback with field cameras, custom-class semantics, and held-out physical volume accuracy remain unverified. The four disposable containers were removed; other containers and volume inventory were unchanged during cleanup. XML/JSON results and logs are retained in `/tmp/opencode/sivera-docker-verify/`.

Execution details, checks, and pending physical validation are recorded in `.agents/plan/2026-10-26-design-agents-alignment.md`. The companion dashboard plan is `../sivera-dashboard/.agents/plan/2026-10-26-depth-contract-alignment.md`. No user database migration is implied by this implementation record.

## 20. Public WHEP repair (31 October 2026)

- Public playback uses `/api/v1/public/cameras/{camera_id}/stream/whep` without an API key. The camera must exist and its stream must be enabled. Stream configuration stays behind the authenticated camera API.
- The old public proxy returned OPTIONS 200, which the dashboard correctly rejected. Capabilities now come from MediaMTX as bodyless 204 with `Accept-Post`, optional `Accept-Patch`, and ICE `Link`. Middleware handles CORS preflight separately; its 200 response is expected.
- POST accepts `application/sdp` and returns 201 SDP. The validated upstream session UUID is rewritten into the public backend `Location`. PATCH accepts ICE fragments and forwards `If-Match`; responses preserve ETag and content type. DELETE maps upstream 200/204/404 to bodyless 204. Requests and responses use the existing bounded MediaMTX service; raw upstream errors and CORS headers are not forwarded.
- Public sessions do not create authenticated `StreamSession` mappings or inherit their ownership/TTL enforcement. Their Location contains the upstream session UUID and must not be logged. The dashboard sends DELETE on component disposal and attempts keepalive cleanup on `pagehide`.
- Verification: 47 public/authenticated route tests passed with mocked database and MediaMTX transport. A real local Chrome session received OPTIONS 204 and POST 201, read Location/ETag through CORS, decoded 1920×1080 video, and received DELETE 204 on in-app navigation. This verifies local playback; segmentation/volume persistence with detected waste and deployment-specific TLS/TURN remain separate checks.

## User management and organization scope

### Current release: one organization

The current scope is a single organization. Keep the model and API simple; do not add speculative organization tables, tenant IDs, organization-switching UI, or claim tenant isolation before multi-organization support is explicitly planned.

### Authentication and roles

The backend's X-API-Key is API/service authentication and must not be mistaken for end-user login, sessions, or role-based access. Any account-management feature must begin by verifying the real authentication provider, principal model, session lifecycle, and API permission contract. Do not invent user CRUD or invitation endpoints.

Suggested product roles, pending implementation: administrator (user/access management and permitted configuration), operator (create measurements and review permitted results), viewer (read-only results/history). Enforce every permission on the backend; frontend route guards and hidden buttons are only usability measures. Distinguish unauthenticated (401), unauthorized (403), expired session, validation errors, and service failures.

Where supported, user administration should provide user search/list, role and active status, a real creation/invitation flow, confirmed role changes and deactivation, and a safeguard against removing the last administrator. Add tests for allowed and denied actions.

### Future multi-organization extension

Centralize authorization and resource-access decisions so organization scoping can be added without duplicating checks across routes. Before multi-organization launch, define ownership for users, cameras, calibration versions, ground models, depth scales, prediction jobs, and output artifacts; introduce schema/API scope with a documented backfill; and test that one organization cannot read, mutate, or download another organization's resources. This is a future migration, not a requirement to add tenant fields now.
