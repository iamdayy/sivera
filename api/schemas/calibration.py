"""Pydantic schemas for Calibration API."""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class CalibrationBase(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    chessboard_cols: int = Field(9, ge=3, le=30)
    chessboard_rows: int = Field(6, ge=3, le=30)
    square_m: float = Field(0.025, gt=0, le=1.0)
    alpha: float = Field(0.0, ge=0.0, le=1.0)
    notes: Optional[str] = None


class CalibrationCreate(CalibrationBase):
    pass


class CalibrationResponse(BaseModel):
    id: UUID
    camera_id: UUID
    version: int
    k_matrix: List[List[float]] = Field(description="Undistorted camera intrinsics; legacy field retains this meaning")
    k_original: Optional[List[List[float]]] = Field(None, description="Original raw-image intrinsics; null for legacy calibrations, never reconstructed by migration")
    k_undist: Optional[List[List[float]]] = Field(None, description="Explicit target intrinsics for undistortion, equivalent to k_matrix for new calibrations")
    resolution_w: Optional[int] = Field(None, description="Calibration image width in pixels")
    resolution_h: Optional[int] = Field(None, description="Calibration image height in pixels")
    alpha: Optional[float] = None
    per_view_errors: Optional[List[float]] = Field(None, description="Per-view reprojection RMS errors in pixels")
    dist_coeffs: List[float]
    rms_reproj: Optional[float] = Field(description="Overall reprojection RMS error in pixels")
    coverage_w: Optional[float] = Field(description="Horizontal chessboard coverage in percent")
    coverage_h: Optional[float] = Field(description="Vertical chessboard coverage in percent")
    chessboard_cols: Optional[int]
    chessboard_rows: Optional[int]
    square_m: Optional[float]
    calib_images: Optional[List[str]]
    notes: Optional[str]
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class CalibrationListResponse(BaseModel):
    items: List[CalibrationResponse]
    total: int


class CalibrationActivateResponse(BaseModel):
    id: UUID
    camera_id: UUID
    version: int
    is_active: bool
    message: str


class UndistortPreviewResponse(BaseModel):
    preview_url: str
    original_size: List[int]
    undistorted_size: List[int]
    k_undist: List[List[float]]
