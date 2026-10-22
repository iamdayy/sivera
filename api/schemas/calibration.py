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
    k_matrix: List[List[float]]
    k_original: Optional[List[List[float]]] = None
    k_undist: Optional[List[List[float]]] = None
    resolution_w: Optional[int] = None
    resolution_h: Optional[int] = None
    alpha: Optional[float] = None
    per_view_errors: Optional[List[float]] = None
    dist_coeffs: List[float]
    rms_reproj: Optional[float]
    coverage_w: Optional[float]
    coverage_h: Optional[float]
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
