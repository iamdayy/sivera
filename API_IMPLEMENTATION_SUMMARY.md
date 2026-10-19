# API Implementation Summary

## Project Structure
```
/home/iamdayy/codes/sivera/
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings
│   │   ├── database.py            # SQLAlchemy async + session
│   │   ├── storage.py             # Local/S3 storage abstraction
│   │   └── security.py            # API key authentication
│   ├── models/
│   │   ├── __init__.py
│   │   └── orm.py                 # SQLAlchemy models (9 tables)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── camera.py              # Camera Pydantic schemas
│   │   ├── calibration.py         # Calibration schemas
│   │   ├── ground_ref.py          # Ground reference schemas
│   │   └── prediction.py          # Prediction schemas
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── cameras.py             # Camera CRUD endpoints
│   │   ├── calibrations.py        # Calibration endpoints
│   │   ├── ground_refs.py         # Ground reference & model endpoints
│   │   └── predictions.py         # Prediction endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── calibration_service.py # Wraps calibrate.py
│   │   ├── ground_model_service.py # Wraps ipm.py
│   │   ├── segment_service.py     # Wraps segment.py
│   │   ├── depth_service.py       # Wraps depth.py
│   │   ├── volume_service.py      # Wraps volume.py
│   │   └── prediction_orchestrator.py # Chains all services
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── celery_app.py          # Celery configuration
│   │   └── tasks/
│   │       ├── __init__.py
│   │       └── predict_task.py    # Async Celery tasks
│   ├── scripts/
│   │   └── migrate_existing.py    # Migration script
│   └── alembic/
│       ├── env.py
│       ├── script.py.mako
│       └── versions/
│           └── 001_initial.py     # Initial migration
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── test_api_imports.py
└── test_api_client.py
```

## Database Schema (9 Tables)
1. **cameras** - Camera registry with location (PostGIS)
2. **calibrations** - Versioned calibration data (K, dist, RMS)
3. **ground_references** - Versioned ground reference points
4. **ground_models** - Computed homography + pose (R, t, H)
5. **depth_scales** - Depth scale calibration (s, b)
6. **prediction_jobs** - Async prediction jobs
7. **prediction_outputs** - Output files (images, NPZ, JSON)
8. **detected_instances** - Per-instance detection results
9. **volume_estimates** - Volume estimates (grid + convex hull)
10. **api_keys** - API key management

## API Endpoints

### Cameras
- `POST /api/v1/cameras` - Create camera
- `GET /api/v1/cameras` - List cameras
- `GET /api/v1/cameras/{id}` - Get camera
- `PATCH /api/v1/cameras/{id}` - Update camera
- `DELETE /api/v1/cameras/{id}` - Delete camera

### Calibrations
- `POST /api/v1/cameras/{id}/calibrations` - Upload chessboard images → calibrate
- `GET /api/v1/cameras/{id}/calibrations` - List calibrations
- `GET /api/v1/cameras/{id}/calibrations/active` - Get active calibration
- `POST /api/v1/cameras/{id}/calibrations/{id}/activate` - Activate calibration
- `GET /api/v1/cameras/{id}/calibrations/{id}/undistort-preview` - Preview

### Ground References
- `POST /api/v1/cameras/{id}/ground-references` - Submit points → compute model
- `GET /api/v1/cameras/{id}/ground-references` - List
- `GET /api/v1/cameras/{id}/ground-references/active` - Get active
- `POST /api/v1/cameras/{id}/ground-references/{id}/activate` - Activate

### Ground Models
- `GET /api/v1/cameras/{id}/ground-models/active` - Get active model
- `GET /api/v1/cameras/{id}/ground-models/{id}/bev-preview` - BEV preview

### Predictions
- `POST /api/v1/cameras/{id}/predict` - Submit prediction job (async)
- `GET /api/v1/cameras/{id}/predict/{job_id}` - Get job status/result
- `GET /api/v1/cameras/{id}/predict/{job_id}/outputs` - List outputs
- `GET /api/v1/cameras/{id}/predict/{job_id}/outputs/{type}` - Download output
- `GET /api/v1/cameras/{id}/predict/{job_id}/instances` - Get instances
- `GET /api/v1/cameras/{id}/predict/{job_id}/volumes` - Get volumes

## Key Features

1. **Versioned Calibrations** - Multiple calibration versions per camera, only one active
2. **Resolution Handling** - K_undist stored and scaled for processing resolution
3. **Async Processing** - Celery workers for long-running predictions
4. **Storage Abstraction** - Local/S3 compatible storage for output files
5. **PostGIS Support** - Spatial queries for camera locations
6. **API Key Auth** - Header-based authentication with scopes
7. **Complete Pipeline** - Segment → IPM → Depth → Lift → Volume → BEV

## Running the API

### 1. Start Infrastructure
```bash
docker-compose up -d postgres redis
```

### 2. Run Migrations
```bash
cd api
alembic upgrade head
```

### 3. Run Migration Script (Optional - for existing data)
```bash
python scripts/migrate_existing.py
```

### 4. Start API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start Worker (separate terminal)
```bash
celery -A api.workers.celery_app worker --loglevel=info --concurrency=2
```

### 6. Monitor with Flower
```bash
# Via docker-compose
docker-compose up -d flower
# Or directly
celery -A api.workers.celery_app flower --port=5555
```

## Environment Variables
Copy `.env.example` to `.env` and configure:
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection
- `MODELS_DIR` - Path to YOLO models
- `STORAGE_LOCAL_ROOT` - Local file storage path
- `VALID_API_KEYS` - Comma-separated API keys for testing

## Testing
```bash
# Test imports
python test_api_imports.py

# Test with TestClient (requires running DB)
python test_api_client.py
```

## Next Steps
1. Add RTSP frame fetching for `source_type=rtsp_frame`
2. Implement proper undistort using original K + dist
3. Add webhook support for job completion notifications
4. Add periodic RTSP polling via Celery Beat
5. Add authentication with JWT tokens
6. Add Prometheus metrics
7. Add OpenAPI documentation examples