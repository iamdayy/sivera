# SIVERA — Secret Configuration and Synchronization Guide (S1)

This document inventories every secret/configuration variable by **function, not value**.
No secret values are printed, stored, or committed here. `.env` is gitignored and must
never be committed.

---

## 1. Secret inventory

| Variable / config | Function | Owner component | Consumers | Source | Required | Sync dependency | Rotation impact |
|---|---|---|---|---|---|---|---|
| `SECRET_KEY` | HMAC key for API-key hashing (DB), RTSP source identity HMAC, WHEP ownership | Backend API | `api/core/security.py` (`hash_key`), `api/services/snapshot_service.py`, `api/routes/streams.py` | env / `.env` | **Yes in production** (DEBUG=false refuses weak/empty) | All API workers must share the same value | Rotating invalidates all DB-hashed API keys (`api_keys.key_hash`); keys must be re-issued. Also invalidates WHEP owner hashes and RTSP source HMACs |
| `VALID_API_KEYS` | Wildcard environment-configured API keys (no DB row, no expiry) | Backend API | `api/core/security.py` (`valid_api_keys`) | env / `.env` | No | Must match client configuration | Changing requires client update; no DB re-hash needed (compared directly) |
| DB `api_keys.key_hash` | Hashed per-key credentials stored in PostgreSQL | PostgreSQL | `api/core/security.py` | Generated at issuance via `generate_api_key()` | Per deployment | Hash = HMAC-SHA256(`SECRET_KEY`, key) | After `SECRET_KEY` rotation, ALL existing rows are invalid → re-issue every key |
| `DATABASE_URL` / `POSTGRES_PASSWORD` | PostgreSQL connection credentials | PostgreSQL service | Backend (`api/core/database.py`), Compose `postgres` service | env / `.env` / Compose | Yes | Compose prod override requires `POSTGRES_PASSWORD`; base compose uses dev default `waste` (dev only) | Requires DB + API restart; update both Compose and `DATABASE_URL` |
| `MEDIAMTX_API_USER` / `MEDIAMTX_API_PASSWORD` | MediaMTX internal API user credentials | MediaMTX container + Backend | `api/services/mediamtx_service.py`, Compose `mediamtx` env (`MTX_AUTHINTERNALUSERS_1_*`) | env / `.env` | Yes when `MEDIAMTX_ENABLED=true` | Backend `.env` and Compose `mediamtx` service env MUST match (`:?` required check in compose) | Requires MediaMTX container restart + API restart; update both sides |
| `MEDIAMTX_READ_USER` / `MEDIAMTX_READ_PASSWORD` | MediaMTX internal read user (RTSP pull for capture) | MediaMTX container + Backend | `api/services/rtsp_capture_service.py` (builds RTSP URL), Compose `MTX_AUTHINTERNALUSERS_2_*` | env / `.env` | Yes when `MEDIAMTX_ENABLED=true` | Same as API password: both sides must match | Same as above |
| RTSP camera credentials (embedded in per-camera `rtsp_url` in DB) | Upstream camera stream auth | Per camera (DB row) | `validate_rtsp_source` (scheme/host/port validation), capture services | Database | Per camera | Not shared with MediaMTX users | Never logged (credential print removed from `cameras.py`); changing a camera URL requires camera re-registration |
| `STORAGE_S3_ACCESS_KEY` / `STORAGE_S3_SECRET_KEY` | Optional S3-compatible storage credentials | Storage backend | `api/core/storage.py` | env / `.env` | Only when `STORAGE_BACKEND=s3` | S3 provider side | Update storage credentials on provider + API restart |
| `CORS_ORIGINS` | Allowed cross-origin origins (not secret) | Backend API | `api/main.py` CORS middleware | env / `.env` | Production yes (prod override requires it) | Dashboard origin must be listed | API restart only |

---

## 2. Development configuration

- Copy `.env.example` to `.env` (gitignored).
- Generate a development `SECRET_KEY`: `python -c "import secrets; print(f'wst_{secrets.token_urlsafe(32)}')"`
- Base `docker-compose.yml` runs with `DEBUG=true` and dev defaults (e.g. `POSTGRES_PASSWORD=waste`). This is dev-only; never use it in production.
- With `DEBUG=true`, the SECRET_KEY startup gate is bypassed (development mode).

## 3. Production configuration

Use `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`.

Required in environment or `.env` (prod override enforces with `:?`):
- `SECRET_KEY` (>=32 chars, unique, strong)
- `POSTGRES_PASSWORD`
- `CORS_ORIGINS` (explicit list)
- `MEDIAMTX_API_PASSWORD`, `MEDIAMTX_READ_PASSWORD` (if MediaMTX enabled)
- `VALID_API_KEYS` or DB-issued keys

The API refuses to start when `DEBUG=false` and `SECRET_KEY` is empty, default,
placeholder, or shorter than 32 chars (`Settings.enforce_secret_key_for_production`).

## 4. Rotation procedure (per secret)

1. Generate a new value out-of-band (never in logs/reports).
2. Update the source of truth (`.env` / secret manager / provider console).
3. **For `SECRET_KEY`**: after rotation, all DB-hashed API keys are invalid.
   Re-issue every API key via the admin API (new `generate_api_key()`), and update
   `VALID_API_KEYS` clients. WHEP owner hashes and RTSP source HMACs also change;
   existing WHEP sessions must be re-established.
4. **For MediaMTX passwords**: update BOTH the backend `.env` AND the Compose
   `mediamtx` service environment, then restart both containers.
5. **For `POSTGRES_PASSWORD`**: update Postgres and `DATABASE_URL`, restart API.
6. Restart/redeploy affected services.
7. Verify (see section 6).

## 5. Restart / redeployment

- Backend API: `docker compose restart api` (or redeploy).
- MediaMTX: `docker compose restart mediamtx` (required after password change).
- Postgres: password change requires Postgres-side update + API restart.
- A running container does NOT pick up `.env` changes automatically — always restart.

## 6. Post-synchronization verification

- API starts successfully with `DEBUG=false` and a strong `SECRET_KEY` (no RuntimeError).
- A client with a NEW API key can authenticate (401 for old keys after SECRET_KEY rotation).
- MediaMTX API is reachable with the new API credentials; RTSP pull works with the new read credentials.
- WHEP signaling works (POST returns 201 SDP + Location).
- No secret value appears in logs, error responses, or reports.

## 7. If synchronization fails

- Do NOT fall back to old credentials silently.
- Revert to the previous consistent set only through a deliberate rollback with the old values in all components simultaneously.
- Record the failure and the rollback in the operator log (without values).

---

## 8. What must never happen

- Never commit `.env` to Git (gitignored in `sivera/` and `sivera-dashboard/`).
- Never print secret values to terminal, logs, reports, or documentation.
- Never reuse one secret value across unrelated components.
- Never let a development configuration silently apply to production.
