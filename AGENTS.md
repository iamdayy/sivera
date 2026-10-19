# AGENTS.md — SIVERA Backend

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

## API contract rules

1. Treat live `/openapi.json`, route declarations, Pydantic schemas, and tests as the source of truth.
2. Routers are currently included directly by `api/main.py`; do not assume `/api/v1`.
3. API-key authentication uses the `X-API-Key` header, with scope checks for camera, calibration, ground-reference, and prediction operations.
4. Keep schemas compatible with the separate dashboard. Coordinate breaking API changes with `iamdayy/sivera-dashboard`.
5. Use UUIDs where models and schemas define UUID identifiers.
6. Use explicit HTTP status codes and stable, actionable error responses.
7. Never expose secrets, stack traces, internal filesystem paths, or RTSP credentials to clients.
8. When adding a field, update relevant schema, persistence model/migration, service logic, tests, and API documentation.
9. Verify multipart fields, response models, and status codes in OpenAPI after route changes.

## Caveats that must not be hidden

- Prediction requires active camera calibration, active ground model/reference, and a depth scale.
- The prediction route advertises `upload`, `rtsp_frame`, and `latest_rtsp`, but the current handler only implements upload and raises `NotImplementedError` for RTSP.
- Prediction currently executes synchronously despite returning HTTP 202 and containing a TODO for Celery dispatch. Do not claim asynchronous processing until the route enqueues work and status transitions are tested.
- Celery components exist, but verify route/task wiring before changing queue semantics.
- The README may contain outdated route-prefix or authentication examples; source and live OpenAPI take precedence.
- Preserve the distinction between grid/voxel volume and convex-hull volume. BEV is not a true 3D reconstruction.

## Computer vision and geometry

- Keep image coordinates, world coordinates, camera intrinsics/extrinsics, depth scale/bias, and unit conversions explicit.
- Do not assume model depth is metric without a calibrated and verified scale.
- Avoid silent coordinate-system changes or hidden unit conversions.
- Validate calibration and ground-reference input before processing.
- Document physical units; volume should be represented in cubic meters (`m³`) and liters (`L`) where applicable.
- Add tests with known geometric fixtures or synthetic points for coordinate transforms and volume calculations.
- Missing or invalid estimates must remain missing/failed, not be represented as zero.

## Database and storage

- Follow existing async database session patterns.
- Use Alembic migrations for schema changes; do not rely on manual production-only modifications.
- Respect transaction boundaries and explicitly roll back failed writes where appropriate.
- Keep artifact access behind `api/core/storage.py`; do not hardcode local or S3 paths in route handlers.
- Ensure artifact URLs match the configured storage/static-serving mechanism.
- Enforce upload size limits and avoid unbounded memory use for large images.
- Never commit real `.env` values, API keys, credentials, model secrets, or camera RTSP credentials.

## Security and deployment

- Review CORS before production. The current app allows all origins and credentials; do not assume this is safe for deployment.
- Validate upload content, size, and accepted formats.
- Keep API keys out of logs and return only safe error details to clients.
- Use least-privilege scopes and preserve key hashing behavior unless deliberately migrating it.
- Keep safe defaults and document required environment settings in `.env.example`.
- Do not accidentally commit generated data, private images, model weights, or secrets.

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

Do not claim tests passed unless they were run. If GPU, model weights, or external services are unavailable, state which integration tests could not run.

## Working process

1. Inspect relevant routes, schemas, services, models, config, and tests.
2. Trace the current request flow before changing behavior.
3. Identify migration and dashboard compatibility implications.
4. Implement a focused change and add regression tests.
5. Run applicable checks and inspect OpenAPI/output behavior.
6. Update `DESIGN.md` or README when architecture/behavior changes.
7. Summarize changed files, checks run, and remaining risks.
