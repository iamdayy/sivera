from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from api.core.config import settings


class CaptureCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)

    chessboard_cols: int = Field(settings.DEFAULT_CHESSBOARD_COLS, ge=3, le=30)
    chessboard_rows: int = Field(settings.DEFAULT_CHESSBOARD_ROWS, ge=3, le=30)
    square_m: float = Field(settings.DEFAULT_SQUARE_M, gt=0, le=1, description="Chessboard square side in meters")
    interval_sec: float = Field(2, ge=1, le=60)
    target_images: int = Field(20, ge=1, le=100, description="Accepted chessboard frames for new sessions")
    max_duration_sec: int = Field(300, ge=1, le=900, description="Wall-clock limit from session creation, including queue time")
    auto_calibrate: bool = Field(False, description="Queue inactive calibration computation after completed/stopped/expired capture; insufficient accepted frames fail computation")


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
    auto_calibrate: bool = False
    selection_version: int = Field(0, description="1 targets accepted frames; legacy 0 targets stored frames, then quality-assesses them at computation")
    accepted: int = Field(0, description="Frames accepted by chessboard, sharpness, and pose selection")
    rejected: int = Field(0, description="Stored frames rejected by quality selection")
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    compute_status: str = Field("none", description="Independent computation state: none, queued, running, done, failed")
    compute_error_code: str | None = None
    compute_requested_at: datetime | None = None
    compute_started_at: datetime | None = None
    compute_completed_at: datetime | None = None
    result_calibration_id: UUID | None = Field(None, description="Inactive calibration UUID after compute succeeds; explicitly activate it via calibrations endpoint")
    attempted: int = Field(description="Attempts started; interrupted attempts need not equal captured + failed_attempts")
    captured: int = Field(description="All stored frames, including rejected ones; maximum 300 per session")
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
    quality_status: str = "unassessed"
    rejection_reason: str | None = None
    sharpness: float | None = None
    corners: list[list[float]] | None = Field(None, description="Detected chessboard corners in original-image pixel coordinates")
    preview_url: str | None = None


class CaptureImageList(BaseModel):
    items: list[CaptureImageResponse]
    total: int
