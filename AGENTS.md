# AGENTS.md: SIVERA Backend

## Mission and boundaries

Maintain the SIVERA backend in this repository (`iamdayy/sivera`). It owns the FastAPI API, persistence, storage, and computer-vision processing. The dashboard is a separate project: `iamdayy/sivera-dashboard`.

Read `DESIGN.md` before substantial changes. Inspect current source and tests before relying on documentation; README examples may be stale.

## Stack and repository layout

The project uses Python/FastAPI, Pydantic schemas/settings, SQLAlchemy async, PostgreSQL, Alembic, Redis/Celery components, OpenCV, NumPy/SciPy, segmentation/depth services, and Docker. Confirm actual dependencies and commands in repository files.

Important paths:
- `api/main.py`
- `api/core/config.py`, `database.py`, `security.py`, `storage.py`
- `api/models/orm.py`
- `api/schemas/`
- `api/routes/`
- `api/services/`
- `api/workers/`
- `api/alembic/`
- `api/scripts/export_openapi.py`, `openapi.json`
- `deploy/mediamtx.yml`, `docker-compose.yml`

## API contract rules

1. Treat live `/openapi.json`, route declarations, Pydantic schemas, and tests as the source of truth.
2. `api/main.py` mounts camera operations explicitly under `/api/v1/cameras`, including streams, snapshots, calibrations, calibration captures, ground references/models, and predictions. `/files`, `/health`, and `/` are outside that prefix. `API_PREFIX` is not used to mount routers; inspect the declarations rather than assuming a configurable prefix.
3. API-key authentication uses the `X-API-Key` header, with scope checks for camera, calibration, ground-reference, and prediction operations.
4. Keep schemas compatible with the separate dashboard. Coordinate breaking API changes with `iamdayy/sivera-dashboard`.
5. Use UUIDs where models and schemas define UUID identifiers.
6. Use explicit HTTP status codes and stable, actionable error responses.
7. Never expose secrets, stack traces, internal filesystem paths, or RTSP credentials to clients.
8. When adding a field, update relevant schema, persistence model/migration, service logic, tests, and API documentation.
9. Verify multipart fields, response models, and status codes in OpenAPI after route changes.
10. `/`, `/health`, `/docs`, `/redoc`, and `/openapi.json` are public; camera operations and `/files` require `X-API-Key`. OpenAPI must match those dependencies. Missing headers currently return 422, invalid keys 401, insufficient scopes 403.
11. Export the reproducible default-settings contract with `python api/scripts/export_openapi.py openapi.json`. The exporter ignores local `.env` and process settings, connects to no services, and contains no deployment credentials. Compare it with `/openapi.json` in a running disposable environment; configuration-dependent defaults can differ at deployment.
12. Prediction `seg_classes` and `return_outputs` are JSON-array strings inside multipart forms, not repeated fields or a JSON request body. RTSP forbids caller-supplied `file`/`timestamp`; uploads require `file`. Preserve omitted `return_outputs` versus `[]`.

## Caveats that must not be hidden

- Prediction pins compatible IDs: active calibration, latest ground model linked to an active reference for that calibration, and active depth scale for that ground model. Public depth-scale creation, reads, activation, test-image upload, and authenticated undistorted previews are available under the ground model. Creation uses at least three camera-axis Z measurements and auto-activates only accepted fits. The first prediction can still be blocked without a compatible scale; prediction never creates it. Fit success is not physical metric certification.
- Depth-scale creation is synchronous 201, not a queued 202 job. It uses a killable CPU inference subprocess with a configurable timeout and one creation inference per API process; busy/timeouts return safe 503 codes. Keep raw-image undistort-before-inference consistent with prediction. `/files` is not an upload endpoint. Use source keys from test-image upload or registered calibration images; never accept arbitrary filesystem paths/URLs or fabricate measurements.
- `upload`, `rtsp_frame`, and `latest_rtsp` are implemented and queued. The API commits a pending job before dispatch to queue `prediction`, returns 202 with `job_id`, and workers persist `processing -> done/failed`. `estimated_duration_sec=30` is a fixed advisory value, not measured progress or an SLA.
- `rtsp_frame` captures when the worker executes. `latest_rtsp` reselects a fresh persisted snapshot at execution and never captures a fallback. Both require an enabled stream and original calibration intrinsics/dimensions. Timestamps record worker reception, not sensor time.
- Capture/compute use persistent outboxes with bounded redispatch; prediction does not. Broker failure returns 503 and marks the prediction failed; stale jobs fail without automatic resume. Keep task claims and completion fences intact.
- Capture and computation have separate statuses. New sessions target accepted frames (`selection_version=1`), retain rejected frames, and store at most 300. Legacy sessions target stored frames (`selection_version=0`). `completed` means the capture target was reached, not successful calibration.
- Capture compute creates inactive calibration and requires explicit activation. Legacy multipart upload calibration remains synchronous and auto-activates. Do not silently unify these behaviors.
- The MediaMTX image is fixed at `bluenviron/mediamtx:1.20.0-ffmpeg`; use `deploy/mediamtx.yml`, not an unrelated root `mediamtx.yml`. Do not source-build or change its version without approval. WHEP signaling success does not prove browser video/ICE works.
- The README may contain outdated route-prefix or authentication examples; source and live OpenAPI take precedence.
- Preserve volume methods: grid sums cell median heights times cell area; `convex_hull` is a 2D footprint-hull area times median height, not a 3D hull. BEV is not a true 3D reconstruction; grid allocation is capped at 1,000,000 cells.

## Computer vision and geometry

- Keep image coordinates, world coordinates, camera intrinsics/extrinsics, depth scale/bias, and unit conversions explicit.
- Do not assume model depth is metric without a calibrated and verified scale.
- Avoid silent coordinate-system changes or hidden unit conversions.
- Validate calibration and ground-reference input before processing.
- Document physical units; volume should be represented in cubic meters (`m³`) and liters (`L`) where applicable.
- Add tests with known geometric fixtures or synthetic points for coordinate transforms and volume calculations.
- Missing or invalid estimates must remain missing/failed, not be represented as zero.
- `k_matrix` remains undistorted K. New calibrations store raw-image `k_original`, explicit `k_undist`, resolution, alpha, and per-view errors. Do not reconstruct original intrinsics for legacy rows or reinterpret `k_matrix`.
- Explicit-calibration predictions accept raw images at exact calibration dimensions and undistort once. Legacy upload predictions without original K require already-undistorted images at registered camera dimensions. Ground-reference image points belong to undistorted coordinates.
- Chessboard square side is meters; corners/sharpness previews use original image pixels. Reprojection errors are pixels, stored coverage is percent. Quality thresholds are heuristics and cannot certify metric accuracy.

## Database and storage

- Follow existing async database session patterns.
- Use Alembic migrations for schema changes; do not rely on manual production-only modifications.
- Respect transaction boundaries and explicitly roll back failed writes where appropriate.
- Keep artifact access behind `api/core/storage.py`; do not hardcode local or S3 paths in route handlers.
- Ensure artifact URLs match the configured storage/static-serving mechanism.
- Enforce upload size limits and avoid unbounded memory use for large images.
- Never commit real `.env` values, API keys, credentials, model secrets, or camera RTSP credentials.
- Migration head is `005_capture_quality_snapshots`. `create_all` at startup does not upgrade existing tables. A database without Alembic history requires a schema/baseline audit; never blindly `stamp head` or reset it. The authorized local reset on 22 October 2026 was a one-off operation, not permission to repeat it.
- API and all workers share one writable storage root/private bucket. Compose defines separate `capture`, `calibration`, and `prediction` queues; prediction also drains legacy `celery`. Per-task async engines use NullPool to avoid cross-fork/event-loop pooled connections.
- Artifact files survive database/camera deletion and can become orphaned after interrupted writes. Retention and physical cleanup are operator tasks. Never silently delete files while migrating metadata.

## Security and deployment

- CORS uses explicit `CORS_ORIGINS`; empty origins reject cross-origin access unless DEBUG enables a wildcard fallback. Compose forces API `DEBUG=true`; override it for production. Signaling responses expose `Location`, `ETag`, `Link`, `Accept-Post`, and `Accept-Patch`.
- Validate upload content, size, and accepted formats.
- Keep API keys out of logs and return only safe error details to clients.
- Use least-privilege scopes and preserve key hashing behavior unless deliberately migrating it.
- Keep safe defaults and document required environment settings in `.env.example`.
- Do not accidentally commit generated data, private images, model weights, or secrets.
- All API/workers must share one stable non-default `SECRET_KEY`, used for database key hashes, WHEP ownership, and RTSP source HMAC. Rotation invalidates old identities and key hashes unless deliberately migrated.
- WHEP OPTIONS/POST/PATCH/DELETE require `camera:read`; PATCH/DELETE also require the same owning key. Use the rewritten backend `Location`, never internal MediaMTX URLs. Do not automatically retry offer POST. ICE/media reachability and codec compatibility need browser testing.
- Use exact RTSP/RTSPS hostname allowlists plus deployment egress restrictions. MediaMTX admin/reader credentials are distinct and never sent to the dashboard. Authenticated image previews need header-bearing fetch/blob URLs; do not put API keys in query strings.
- Ground-reference/model/depth-scale routers enforce camera-bound key restrictions. Known legacy gaps: database-key `expires_at` is not checked by authentication; generic `/files` fallback branches do not enforce a domain read scope/registration for every possible key shape. Do not claim complete tenant isolation or expiry enforcement. Coordinate focused fixes and regression tests before a multi-tenant production rollout.

## Code quality

- Follow existing Python style and async conventions.
- Keep route handlers focused and move domain/vision logic into service functions when appropriate.
- Maintain clear route/schema/service responsibilities.
- Avoid broad refactors unrelated to the task.
- Handle expected validation errors explicitly; log unexpected diagnostics server-side.
- Add type annotations to new public functions and maintain dependency-injection patterns.

## Testing and verification

Use only commands that exist in the repository and environment. Confirm installed dependencies and test paths before assuming every command is available.

As relevant:
- test schema and route validation;
- test authentication and scope enforcement;
- test database behavior and migrations using a disposable database;
- test calibration, depth, ground mapping, and volume services with deterministic fixtures;
- test upload and artifact storage behavior;
- verify `/health` and `/openapi.json` in a running test environment.
- verify WHEP SDP/ICE content types, Location/ETag/If-Match, ownership, and session cleanup;
- test accepted/rejected selection, source/resolution changes, capture stop/deadline/crash, compute idempotency/inactive result, snapshots/freshness, prediction queue claims, selected outputs, and missing estimates;
- run migration upgrade/downgrade only against disposable databases. Integration tests can mutate schema/paths and kill a worker child when explicitly enabled; never point their environment variables at user services.

Do not claim tests passed unless they were run. If GPU, model weights, or external services are unavailable, state which integration tests could not run.

## Working process

1. Inspect relevant routes, schemas, services, models, config, and tests.
2. Trace the current request flow before changing behavior.
3. Identify migration and dashboard compatibility implications.
4. Implement a focused change and add regression tests.
5. Run applicable checks and inspect OpenAPI/output behavior.
6. Update `DESIGN.md`, this file, relevant README sections, and the OpenAPI snapshot when architecture/behavior/contracts change. Keep frontend implementation in `../sivera-dashboard`; integration planning belongs in its `.agents/plan/` without overwriting historical plans.
7. Summarize changed files, checks run, and remaining risks.
