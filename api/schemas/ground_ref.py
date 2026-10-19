"""Pydantic schemas for Ground Reference and Ground Model API."""
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