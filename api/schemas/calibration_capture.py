from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.core.config import settings


class CaptureCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)

    chessboard_cols: int = Field(settings.DEFAULT_CHESSBOARD_COLS, ge=2, le=30)
    chessboard_rows: int = Field(settings.DEFAULT_CHESSBOARD_ROWS, ge=2, le=30)
    square_m: float = Field(settings.DEFAULT_SQUARE_M, gt=0, le=1, description="Chessboard square side in meters")
    interval_sec: float = Field(2, ge=1, le=60)
    target_images: int = Field(20, ge=1, le=100, description="Stored frames, not quality-selected images in phase 2")
    max_duration_sec: int = Field(300, ge=1, le=900, description="Wall-clock limit from session creation, including queue time")
    auto_calibrate: Literal[False] = Field(False, description="Automatic computation is not available until phase 3")


class CaptureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: UUID = Field(validation_alias="id")
    camera_id: UUID
    status: Literal["queued", "running", "stopping", "completed", "stopped", "expired", "failed"]
    chessboard_cols: int
    chessboard_rows: int
    square_m: float
    interval_sec: float
    target_images: int
    max_duration_sec: int
    auto_calibrate: Literal[False] = False
    attempted: int
    captured: int
    failed_attempts: int
    resolution_w: int | None
    resolution_h: int | None
    error_code: str | None
    finish_reason: str | None
    created_at: datetime
    deadline_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    heartbeat_at: datetime | None
    last_capture_at: datetime | None


class CaptureImageResponse(BaseModel):
    id: UUID
    sequence: int
    width: int
    height: int
    size_bytes: int
    captured_at: datetime
    download_url: str


class CaptureImageList(BaseModel):
    items: list[CaptureImageResponse]
    total: int
