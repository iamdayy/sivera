"""Application configuration using Pydantic Settings."""
import json
from pathlib import Path
from typing import List, Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Waste Volume Estimation API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_PREFIX: str = "/api/v1"
    
    # Security
    API_KEY_HEADER: str = "X-API-Key"
    VALID_API_KEYS: str = ""
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week
    CORS_ORIGINS: str = ""  # comma-separated origins; empty = secure default (see main.py)
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://waste:waste@localhost:5432/waste_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 300  # 5 minutes
    
    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = 10  # maximum upload size in megabytes
    
    # File Storage
    STORAGE_BACKEND: str = "local"  # "local" | "s3"
    STORAGE_LOCAL_ROOT: str = "/data/waste-api"
    STORAGE_S3_BUCKET: str = ""
    STORAGE_S3_REGION: str = "us-east-1"
    STORAGE_S3_ACCESS_KEY: str = ""
    STORAGE_S3_SECRET_KEY: str = ""
    STORAGE_S3_ENDPOINT_URL: Optional[str] = None
    
    # Models
    MODELS_DIR: str = "/models"
    DEFAULT_SEG_MODEL: str = "yolo11s-seg.pt"
    DEFAULT_DEPTH_MODEL: str = "yolo26n-depth.pt"
    
    # Processing defaults
    DEFAULT_SEG_CONF: float = 0.25
    DEFAULT_SEG_IOU: float = 0.7
    DEFAULT_SEG_IMGSZ: int = 1280
    DEFAULT_DEPTH_IMGSZ: int = 768
    DEFAULT_VOLUME_CELL_M: float = 0.05
    DEFAULT_VOLUME_MIN_PTS_PER_CELL: int = 2
    DEFAULT_BEV_PPM: int = 200
    DEFAULT_BEV_MARGIN_M: float = 1.0
    DEFAULT_DEPTH_CLIP_MIN_M: float = 0.1
    DEFAULT_DEPTH_CLIP_MAX_M: float = 50.0
    DEPTH_SCALE_TIMEOUT_SEC: int = Field(120, ge=10, le=600)
    DEPTH_SCALE_MAX_MEAN_RESIDUAL_M: float = Field(0.25, gt=0, allow_inf_nan=False)
    DEPTH_SCALE_MAX_RESIDUAL_M: float = Field(0.5, gt=0, allow_inf_nan=False)
    DEPTH_SCALE_MAX_RELATIVE_RESIDUAL: float = Field(0.1, gt=0, le=1, allow_inf_nan=False)
    
    # Calibration defaults
    DEFAULT_CHESSBOARD_COLS: int = 9
    DEFAULT_CHESSBOARD_ROWS: int = 6
    DEFAULT_SQUARE_M: float = 0.025
    MIN_CALIB_IMAGES: int = 10
    RECOMMENDED_CALIB_IMAGES: int = 20
    
    # MediaMTX (CCTV Streaming)
    MEDIAMTX_HOST: str = "localhost"
    MEDIAMTX_RTSP_PORT: int = 8554
    MEDIAMTX_WEBRTC_PORT: int = 8889
    MEDIAMTX_HLS_PORT: int = 8888
    MEDIAMTX_API_PORT: int = 9997
    MEDIAMTX_RTMP_PORT: int = 1935
    MEDIAMTX_ENABLED: bool = False
    MEDIAMTX_API_USER: str = "sivera_api"
    MEDIAMTX_API_PASSWORD: SecretStr = SecretStr("")
    MEDIAMTX_READ_USER: str = "sivera_reader"
    MEDIAMTX_READ_PASSWORD: SecretStr = SecretStr("")
    MEDIAMTX_ALLOWED_RTSP_HOSTS: str = ""
    MEDIAMTX_RECONCILE_INTERVAL: int = Field(30, ge=5, le=3600)
    MEDIAMTX_SESSION_TTL_SEC: int = Field(86400, ge=60, le=604800)
    MEDIAMTX_MAX_SDP_BYTES: int = Field(65536, ge=1024, le=1048576)
    MEDIAMTX_WEBRTC_ADDITIONAL_HOSTS: str = "127.0.0.1"
    # Stream settings
    STREAM_TIMEOUT: int = Field(10, ge=1, le=60)
    STREAM_RETRY_ATTEMPTS: int = Field(3, ge=1, le=5)
    STREAM_CAPTURE_TIMEOUT: int = Field(15, ge=1, le=60)
    CAPTURE_MAX_FRAME_BYTES: int = Field(10485760, ge=65536, le=52428800)
    CAPTURE_MAX_PIXELS: int = Field(16777216, ge=65536, le=67108864)
    CAPTURE_MAX_CONSECUTIVE_FAILURES: int = Field(3, ge=1, le=5)
    CAPTURE_HEARTBEAT_TIMEOUT_SEC: int = Field(45, ge=15, le=120)
    CAPTURE_QUEUE_TIMEOUT_SEC: int = Field(120, ge=30, le=600)
    CAPTURE_RECONCILE_INTERVAL_SEC: int = Field(5, ge=1, le=30)
    CALIB_CAPTURE_MIN_SHARPNESS: float = Field(80, ge=0, le=10000)
    CALIB_COMPUTE_TIMEOUT_SEC: int = Field(300, ge=30, le=600)

    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        enable_decoding=False,
    )

    @property
    def valid_api_keys(self) -> List[str]:
        """Return configured API keys from comma-separated or JSON-list input."""
        value = self.VALID_API_KEYS.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(key).strip() for key in parsed if str(key).strip()]
        return [key.strip() for key in value.split(",") if key.strip()]


settings = Settings()

# Convenience paths (directories are created at application startup, not import time)
MODELS_DIR = Path(settings.MODELS_DIR)
STORAGE_ROOT = Path(settings.STORAGE_LOCAL_ROOT)
