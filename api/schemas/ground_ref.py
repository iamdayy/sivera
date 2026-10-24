"""Pydantic schemas for Ground Reference, Ground Model, and Depth Scale API."""
from datetime import datetime
from typing import Optional, List, Literal
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


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


class DepthMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    x: int = Field(ge=0, strict=True, description="Integer column in the undistorted preview, not CSS pixels")
    y: int = Field(ge=0, strict=True, description="Integer row in the undistorted preview, not CSS pixels")
    distance_m: float = Field(gt=0, strict=True, description="Camera-axis Z in meters, NOT Euclidean rangefinder distance")


class DepthScaleCreate(BaseModel):
    """Fit an additional linear correction to model depth using camera-axis Z measurements."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    
    test_image: str = Field(min_length=1, max_length=500, description="Relative storage key returned by POST .../depth-scales/test-images, or an image registered in this calibration's calib_images. /files is download-only; URLs and filesystem paths are not accepted.")
    depth_model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.pt$", max_length=200, description="Installed depth model filename under MODELS_DIR; no downloading or paths")
    known_distances: List[DepthMeasurement] = Field(min_length=3, max_length=100, description="At least three distinct undistorted pixels with camera-axis Z measurements spanning different depths")
    clip_min_m: float = Field(0.1, ge=0.01, le=10.0)
    clip_max_m: float = Field(50.0, ge=5.0, le=200.0)

    @field_validator("test_image")
    @classmethod
    def storage_key(cls, value: str) -> str:
        if "\\" in value or ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("depth_scale_invalid_image_key")
        return value

    @model_validator(mode="after")
    def validate_measurements(self) -> "DepthScaleCreate":
        if self.clip_min_m >= self.clip_max_m:
            raise ValueError("depth_scale_invalid_clip_range")
        if len({(point.x, point.y) for point in self.known_distances}) != len(self.known_distances):
            raise ValueError("depth_scale_duplicate_pixels")
        if any(not self.clip_min_m <= point.distance_m <= self.clip_max_m for point in self.known_distances):
            raise ValueError("depth_scale_measurement_outside_clip_range")
        return self


class DepthScaleImageResponse(BaseModel):
    image_id: UUID
    test_image: str = Field(description="Source storage key to pass unchanged to depth-scale creation")
    image_url: str = Field(description="Authenticated undistorted PNG preview; fetch with X-API-Key")
    width: int
    height: int
    coordinate_system: Literal["undistorted_pixels"] = "undistorted_pixels"
    distance_type: Literal["camera_axis_z_m"] = "camera_axis_z_m"


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
