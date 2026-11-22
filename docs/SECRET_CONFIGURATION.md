# SIVERA — Secret Configuration and Synchronization Guide (S1)

This document inventories every secret/configuration variable by **function, not value**.
No secret values are printed, stored, or committed here. `.env` is gitignored and must
never be committed.

---

## 1. Secret inventory

| Variable / config | Function | Owner component | Consumers | Source | Required | Sync dependency | Rotation impact |
|---|---|---|---|---|---|---|---|
| `SECRET_KEY` | HMAC key for API-key hashing (DB), RTSP source identity HMAC, WHEP ownership | Backend API | `api/core/security.py` (`hash_key`), `api/services/snapshot_service.py`, `api/routes/streams.py` | env / `.env` | **Yes in production** (DEBUG=false refuses weak/empty) | All API workers must share the same value | Rotating invalidates all DB-hashed API keys (`api_keys.key_hash`), WHEP `owner_hash` rows, and every stored RTSP `source_hash`. See section 4 |
| `VALID_API_KEYS` | Wildcard environment-configured API keys (no DB row, no expiry) | Backend API | `api/core/security.py` (`valid_api_keys`) | env / `.env` | No | Must match client configuration | Compared **in plaintext** (`security.py:30`), independent of `SECRET_KEY`; changing requires client + env update only |
| DB `api_keys.key_hash` | Hashed per-key credentials stored in PostgreSQL (`String(64)`, unique, indexed — `api/models/orm.py:553`, migration `001_initial.py:206`) | PostgreSQL | `api/core/security.py` (`verify_api_key` line 42, `hash_key` line 56) | Inserted directly into DB (no HTTP admin route exists; `generate_api_key()` at `security.py:68` is only referenced by tests) | Per deployment | Hash = HMAC-SHA256(`SECRET_KEY`, key) — verified empirically: different SECRET_KEY → different hash | After `SECRET_KEY` rotation, ALL existing rows fail the `key_hash == hash_key(key)` lookup → must re-insert every key under the new SECRET_KEY |
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
- Base `docker-compose.yml` runs with `DEBUG=true` and dev defaults (e.g. `POSTGRES_PASSWORD=waste`, postgres published `5432`, api published `8000` on all interfaces). This is dev-only; never use it in production.
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
placeholder, or shorter than 32 chars (`Settings.enforce_secret_key_for_production`,
called in lifespan `api/main.py:41` before any request is served).

**Compose port exposure (fixed):** the prod override uses the `!override` tag on
`ports` (requires **Docker Compose >= 2.24.0**). Verified with
`docker compose -f docker-compose.yml -f docker-compose.prod.yml config` (Compose v5.6.1):
- `postgres`: `ports: None` — no published host port in production.
- `api`: only `127.0.0.1:8000:8000` — behind a reverse proxy, not on all interfaces.
- `mediamtx`: unchanged (RTSP/RTMP/WHIP/HTTP API on `127.0.0.1`, ICE UDP `8189` published — required for WebRTC).
- Inter-container traffic uses the default Docker network; no published port is needed for it.
- Without `!override`, Compose **merges** port lists, so base `5432` and `8000` stay exposed even with the prod override — that defect is why `!override` is mandatory here.
- Prod merge without the required env vars fails closed (rc=1, "required variable ... is missing").

---

## 4. SECRET_KEY dependency trace (lifecycle audit)

Traced through actual code paths, not just HMAC usage.

| Komponen | Bergantung pada SECRET_KEY | Dampak rotasi | Pemulihan |
|---|---|---|---|
| **API key (DB)** | Yes — `hash_key()` = HMAC-SHA256(`SECRET_KEY`, key) at `security.py:56-65`. Stored as `api_keys.key_hash` (`orm.py:553`). Verified at `verify_api_key` `security.py:42` and `optional_scope` `security.py:119` via `WHERE key_hash == hash_key(key) AND is_active` | ALL DB rows stop matching → every DB-issued key returns 401. `expires_at`/`is_active` unchanged but irrelevant. `VALID_API_KEYS` keys are NOT affected (plaintext compare, `security.py:30`) | No HTTP admin route exists (`generate_api_key()` `security.py:68` is test-only). Operator inserts new rows directly: generate `(plain, hash_key(plain))` under the NEW SECRET_KEY (one-off script) or `INSERT` the hash; distribute plain keys to clients. Deactivate/retire old rows with `UPDATE api_keys SET is_active=false` (direct DB; no route mutates `ApiKey`) |
| **WHEP session** | Yes — `owner_hash=hash_key(X-API-Key)` stored in `stream_sessions.owner_hash` at creation (`streams.py:175`); verified on PATCH/DELETE via `hmac.compare_digest(session.owner_hash, hash_key(header))` (`streams.py:199`). Hash is **stored**, not recomputed from a token | Every live `stream_sessions` row fails the compare → PATCH/DELETE return 404 (`stream_session_not_found`); clients must POST a new offer. Rows also expire at `expires_at` (`MEDIAMTX_SESSION_TTL_SEC`, default 86400s, max 604800s, `config.py:121`) and are purged by `stream_service.py:39-41` | No re-issue possible for old rows — clients re-POST `/stream/whep` with a valid (re-issued) API key; new rows get fresh `owner_hash`. Public WHEP (`public_stream.py`) has no `StreamSession` ownership mapping and is unaffected |
| **RTSP source identity** | Yes — `source_hash(rtsp_url)` = HMAC-SHA256(`SECRET_KEY`, rtsp_url) at `snapshot_service.py:20-21`. Stored in DB columns: `camera_snapshots.source_hash` (`orm.py:257`), `calibration_capture_sessions.source_hash` (nullable, `orm.py:145`), `prediction_jobs.source_hash` (`orm.py:464`), `ground_reference_sources.source_hash` (`orm.py:419`, migration 008). Recomputed on each use and compared (`snapshot_service.py:36,73`, `calibration_capture_service.py:184-189,243-248`, `calibration_compute_service.py:96-99,159-162`, `ground_reference_source_service.py:57`, `prediction_runner_service.py:68,91`, `auto_prediction_service.py:51-62`) | Stored hashes no longer match recomputed ones → `snapshot_source_changed`, `ground_reference_source_changed`, capture-session source mismatch, prediction job source mismatch errors. New captures overwrite/refresh `source_hash` where the code allows (capture service rewrites `session.source_hash`); snapshots are keyed by the new hash going forward | No signature is sent to MediaMTX — the hash is internal identity only. After rotation + API restart, new snapshots/captures/predictions compute and store the new hash automatically; ground-reference sources fail `require_source_context` until re-registered from a fresh snapshot. Old snapshot rows become unreachable via `latest_snapshot` (hash-keyed lookup) — treat as historical |

Additional `SECRET_KEY`-independent notes: `VALID_API_KEYS` has no DB row and no
expiry; DB keys expire at `expires_at <= now` UTC (`security.py:46`).

---

## 5. Safe SECRET_KEY rotation procedure (operational)

Code-supported steps are marked **[code]**; steps needing operator action are **[operator]**.
No zero-downtime is claimed: rotation is a **disruptive, coordinated maintenance window**.

1. **Preparation [operator]**
   - Generate the new `SECRET_KEY` out-of-band (e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`); never print/log it.
   - Announce a maintenance window; expect all DB API keys, WHEP sessions, and RTSP-source-bound operations to fail during the switch.
   - Prepare the re-issue script for API keys: for each key name/scope/camera, call `generate_api_key()` under the NEW `SECRET_KEY` and stage the `INSERT` statements. Keep the plain keys in the secret manager only.
2. **Backup [operator]** — no production DB changes are made by this guide; backup is operator-run:
   - `pg_dump` (or snapshot) of `api_keys`, `stream_sessions`, `camera_snapshots`, `calibration_capture_sessions`, `prediction_jobs`, `ground_reference_sources` tables.
   - Copy current `.env` / secret-manager version (encrypted) for rollback.
3. **Impact acknowledgment [code-verified, section 4]**: rotation invalidates all DB `api_keys.key_hash` rows (401), all `stream_sessions.owner_hash` (404 on PATCH/DELETE), and all stored RTSP `source_hash` values. `VALID_API_KEYS` is unaffected.
4. **Order of changes [operator]**
   1. Stage new API-key rows (hashes computed with the NEW `SECRET_KEY`) but do not deactivate old rows yet.
   2. Update `SECRET_KEY` in the secret manager / `.env` for the API service.
   3. Restart the API (`docker compose restart api` or redeploy). Old DB keys now 401; staged new keys work immediately after insert — insert them in the same window (direct DB write, no admin route).
   4. Restart API once more if env is injected at container start (a running container does not pick up `.env` changes automatically).
5. **Service restarts [operator]**: only the API consumes `SECRET_KEY`. MediaMTX and Postgres do not need restarts for a `SECRET_KEY` rotation (only for their own passwords). If any runtime worker processes share the env (they read `settings` at import), restart them too.
6. **Re-issue credentials/tokens [operator + client]**
   - Distribute new plain API keys to clients; clients using DB keys get 401 until they switch. Clients using `VALID_API_KEYS` are unaffected.
   - WHEP clients re-POST `/stream/whep` (new `owner_hash` row). No token re-issue exists for old sessions.
   - RTSP sources need no credential change; fresh snapshots/captures recompute `source_hash`. Ground-reference sources bound to old hashes must be re-registered from fresh snapshots.
7. **Verification [operator, checklist in section 7]**: API starts with `DEBUG=false`; new key authenticates; old DB key returns 401; WHEP POST 201 + PATCH/DELETE 204 with the new key; snapshot/predict RTSP flows work; no secret values in logs.
8. **Rollback [operator]** if verification fails: restore the OLD `SECRET_KEY` in all API instances simultaneously, restart API, restore old API-key rows from backup (or rely on `VALID_API_KEYS` as the bridge). Do NOT mix old and new `SECRET_KEY` across API replicas — hash lookups would fail on some workers only. Record the rollback in the operator log without values.

## 6. Rotation procedure for other secrets (per secret)

1. Generate a new value out-of-band (never in logs/reports).
2. Update the source of truth (`.env` / secret manager / provider console).
3. **For MediaMTX passwords**: update BOTH the backend `.env` AND the Compose `mediamtx` service environment, then restart both containers.
4. **For `POSTGRES_PASSWORD`**: update Postgres and `DATABASE_URL`, restart API.
5. **For `VALID_API_KEYS`**: update env + all clients; no DB re-hash (plaintext compare).
6. Restart/redeploy affected services.
7. Verify (see section 7).

## 7. Restart / redeployment

- Backend API: `docker compose restart api` (or redeploy).
- MediaMTX: `docker compose restart mediamtx` (required after password change).
- Postgres: password change requires Postgres-side update + API restart.
- A running container does NOT pick up `.env` changes automatically — always restart.

## 8. Post-synchronization verification

- API starts successfully with `DEBUG=false` and a strong `SECRET_KEY` (no RuntimeError).
- A client with a NEW API key can authenticate (401 for old DB keys after SECRET_KEY rotation).
- MediaMTX API is reachable with the new API credentials; RTSP pull works with the new read credentials.
- WHEP signaling works (POST returns 201 SDP + Location; PATCH/DELETE 204 with owning key).
- Merged compose config shows `postgres ports: None`, api only `127.0.0.1:8000`
  (`docker compose -f docker-compose.yml -f docker-compose.prod.yml config`).
- No secret value appears in logs, error responses, or reports.

## 9. If synchronization fails

- Do NOT fall back to old credentials silently.
- Revert to the previous consistent set only through a deliberate rollback with the old values in all components simultaneously.
- Record the failure and the rollback in the operator log (without values).

---

## 10. What must never happen

- Never commit `.env` to Git (gitignored in `sivera/` and `sivera-dashboard/`).
- Never print secret values to terminal, logs, reports, or documentation.
- Never reuse one secret value across unrelated components.
- Never let a development configuration silently apply to production.
- Never run the base `docker-compose.yml` alone in production (DEBUG=true, postgres `5432` and api `8000` exposed on all interfaces).
