"""Pydantic schemas for Ground Reference, Ground Model, and Depth Scale API."""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class GroundReferenceBase(BaseModel):
    image_points: List[List[float]] = Field(..., min_length=4)
    world_points: List[List[float]] = Field(..., min_length=4)
    world_units: str = Field("meter", pattern="^(meter|metre)$")
    description: Optional[str] = None


class GroundReferenceCreate(GroundReferenceBase):
    pass


class GroundReferenceResponse(BaseModel):
    id: UUID
    camera_id: UUID
    calibration_id: UUID
    version: int
    image_points: List[List[float]]
    world_points: List[List[float]]
    world_units: str
    description: Optional[str]
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class GroundReferenceListResponse(BaseModel):
    items: List[GroundReferenceResponse]
    total: int


class GroundModelResponse(BaseModel):
    id: UUID
    camera_id: UUID
    calibration_id: UUID
    ground_ref_id: UUID
    R_matrix: List[List[float]]
    t_vector: List[float]
    H_i2w: List[List[float]]
    H_w2i: List[List[float]]
    cam_height: Optional[float]
    pitch_down_deg: Optional[float]
    reproj_err_px: Optional[float]
    bev_ppm: int
    bev_margin_m: float
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class BEVPreviewResponse(BaseModel):
    bev_image_url: str
    bev_mask_url: str
    bev_overlay_url: str
    bev_info: dict


# Depth Scale schemas
class DepthScaleCreate(BaseModel):
    """Create depth scale from a test image with known distance measurements."""
    model_config = ConfigDict(extra="forbid")
    
    test_image: str = Field(..., description="Storage key of test image (uploaded via /files or calibration capture)")
    depth_model: str = Field(..., description="Depth model name used (e.g., 'yolo26n-depth.pt')")
    known_distances: List[dict] = Field(..., description="List of {x, y, distance_m} for known distance points in image coordinates (undistorted)")
    clip_min_m: float = Field(0.1, ge=0.01, le=10.0)
    clip_max_m: float = Field(50.0, ge=5.0, le=200.0)


class DepthScaleResponse(BaseModel):
    id: UUID
    camera_id: UUID
    ground_model_id: UUID
    depth_model: str
    scale_s: float = Field(description="Scale factor: metric_depth = raw_depth * scale_s + offset_b")
    offset_b: float = Field(description="Bias/offset in meters")
    residuals: Optional[List[float]] = Field(None, description="Per-point residuals in meters")
    mean_residual_m: Optional[float] = Field(None, description="Mean absolute residual in meters")
    max_residual_m: Optional[float] = Field(None, description="Max absolute residual in meters")
    clip_min_m: float
    clip_max_m: float
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class DepthScaleListResponse(BaseModel):
    items: List[DepthScaleResponse]
    total: int