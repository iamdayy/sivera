"""Application configuration using Pydantic Settings."""
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


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
    VALID_API_KEYS: List[str] = Field(default_factory=list)
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week
    
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
    
    # Calibration defaults
    DEFAULT_CHESSBOARD_COLS: int = 9
    DEFAULT_CHESSBOARD_ROWS: int = 6
    DEFAULT_SQUARE_M: float = 0.025
    MIN_CALIB_IMAGES: int = 10
    RECOMMENDED_CALIB_IMAGES: int = 20
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()

# Convenience paths
MODELS_DIR = Path(settings.MODELS_DIR)
STORAGE_ROOT = Path(settings.STORAGE_LOCAL_ROOT)
# Only create directory if it exists or we have permissions
try:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
except PermissionError:
    # Fallback to local directory
    STORAGE_ROOT = Path("./data/waste-api")
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)