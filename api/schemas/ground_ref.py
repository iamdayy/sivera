"""Pydantic schemas for Ground Reference, Ground Model, and Depth Scale API."""
from datetime import datetime
from typing import Annotated, Optional, List, Literal
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from api.core.config import settings


GroundCoordinate = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class GroundReferenceSourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: UUID
    calibration_id: UUID = Field(description="Active calibration that will undistort this snapshot")


class GroundReferenceSourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    camera_id: UUID
    snapshot_id: UUID
    calibration_id: UUID
    captured_at: datetime = Field(description="Snapshot reception time, not sensor time")
    width: int
    height: int
    coordinate_system: Literal["undistorted_pixels"] = "undistorted_pixels"
    image_url: str = Field(description="Authenticated, persisted undistorted PNG; fetch with X-API-Key")


class GroundReferenceBase(BaseModel):
    image_points: List[tuple[GroundCoordinate, GroundCoordinate]] = Field(..., min_length=4, max_length=100)
    world_points: List[tuple[GroundCoordinate, GroundCoordinate]] = Field(..., min_length=4, max_length=100)
    world_units: str = Field("meter", pattern="^(meter|metre)$")
    description: Optional[str] = None


class GroundReferenceCreate(GroundReferenceBase):
    source_id: Optional[UUID] = None

    @model_validator(mode="after")
    def matching_pairs(self) -> "GroundReferenceCreate":
        if len(self.image_points) != len(self.world_points):
            raise ValueError("Point counts must match")
        return self


class GroundReferenceResponse(BaseModel):
    id: UUID
    camera_id: UUID
    calibration_id: UUID
    source_id: Optional[UUID] = Field(None, description="Persisted camera source; null for legacy references")
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
    bev_mask_url: Optional[str] = None
    bev_overlay_url: str
    bev_info: dict
    source_id: Optional[UUID] = Field(None, description="Bound reference source used by this preview, if any")
    has_mask: bool = Field(False, description="True only when a real mask was supplied; no synthetic segmentation")


class DepthMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    x: int = Field(ge=0, strict=True, description="Integer column in the undistorted preview, not CSS pixels")
    y: int = Field(ge=0, strict=True, description="Integer row in the undistorted preview, not CSS pixels")
    distance_m: float = Field(gt=0, strict=True, description="Camera-axis Z in meters, NOT Euclidean rangefinder distance")


class DepthScaleCreate(BaseModel):
    """Fit an additional linear correction to model depth using camera-axis Z measurements.
    
    Only `known_distances` (min 3 points on undistorted preview) is required.
    Two modes:
    1. Legacy: Provide `test_image` (or default) + `known_distances` — runs YOLO + fits in one call
    2. Two-step: Provide `depth_image_id` (from /preview) + `known_distances` — uses stored depth from preview
    
    `test_image` defaults to first calibration image (legacy mode).
    `depth_model` defaults to DEFAULT_DEPTH_MODEL.
    `clip_min_m`/`clip_max_m` default to 0.1m / 50m.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Two-step flow: depth_image_id from /preview endpoint
    depth_image_id: Optional[UUID] = Field(None, description="Temporary ID from /preview step; if provided, test_image is ignored")
    # Legacy flow: test_image source key
    test_image: Optional[str] = Field(None, min_length=1, max_length=1024, description="Source key returned by test-image upload or registered in this calibration; omitted uses its first image")
    depth_model: str = Field(default_factory=lambda: settings.DEFAULT_DEPTH_MODEL, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.pt$")
    clip_min_m: float = Field(0.1, gt=0)
    clip_max_m: float = Field(50.0, gt=0)
    
    known_distances: List[DepthMeasurement] = Field(min_length=3, max_length=100, description="At least three distinct undistorted pixels with camera-axis Z measurements spanning different depths")

    @model_validator(mode="after")
    def validate_mode(self) -> "DepthScaleCreate":
        if self.depth_image_id is not None and self.test_image is not None:
            raise ValueError("Provide either depth_image_id (two-step) or test_image (legacy), not both")
        if self.depth_image_id is None and self.test_image is None:
            # test_image will default to first calibration image in legacy flow
            pass
        return self

    @field_validator("test_image")
    @classmethod
    def storage_key(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and (":" in value or "\\" in value or any(part in {"", ".", ".."} for part in value.split("/"))):
            raise ValueError("Use a registered relative source key")
        return value

    @model_validator(mode="after")
    def measurement_bounds(self) -> "DepthScaleCreate":
        if self.clip_min_m >= self.clip_max_m:
            raise ValueError("clip_min_m must be less than clip_max_m")
        pixels = {(point.x, point.y) for point in self.known_distances}
        if len(pixels) != len(self.known_distances):
            raise ValueError("Measurement pixels must be distinct")
        if any(not self.clip_min_m <= point.distance_m <= self.clip_max_m for point in self.known_distances):
            raise ValueError("Measurements must lie within the depth clipping bounds")
        return self


class DepthScaleImageResponse(BaseModel):
    image_id: UUID
    test_image: str = Field(description="Source storage key to pass unchanged to depth-scale creation")
    image_url: str = Field(description="Authenticated undistorted PNG preview; fetch with X-API-Key")
    width: int
    height: int
    coordinate_system: Literal["undistorted_pixels"] = "undistorted_pixels"
    distance_type: Literal["camera_axis_z_m"] = "camera_axis_z_m"


class DepthScalePreviewCreate(BaseModel):
    """Run YOLO depth inference on a test image and return depth visualization.
    
    `test_image` defaults to first calibration image.
    `depth_model` defaults to DEFAULT_DEPTH_MODEL.
    """
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    test_image: Optional[str] = Field(None, min_length=1, max_length=1024, description="Source key returned by test-image upload or registered in this calibration; omitted uses its first image")
    depth_model: str = Field(default_factory=lambda: settings.DEFAULT_DEPTH_MODEL, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.pt$")

    @field_validator("test_image")
    @classmethod
    def storage_key(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and (":" in value or "\\" in value or any(part in {"", ".", ".."} for part in value.split("/"))):
            raise ValueError("Use a registered relative source key")
        return value


class DepthScalePreviewResponse(BaseModel):
    """Response from depth preview endpoint."""
    model_config = ConfigDict(from_attributes=True)

    depth_image_id: UUID = Field(description="Temporary ID to reference the computed depth array in the create step")
    depth_visualization_url: str = Field(description="Authenticated depth colormap PNG; fetch with X-API-Key")
    width: int
    height: int
    depth_stats: dict = Field(description="Raw depth statistics: min, max, mean")
    model_used: str
    imgsz: int


class DepthModelIdentity(BaseModel):
    sha256: str
    ultralytics_version: str
    torch_version: str
    opencv_version: str
    preprocessing_version: str
    depth_convention: Literal["camera_axis_z"]
    imgsz: int


class DepthScaleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    camera_id: UUID
    ground_model_id: UUID
    depth_model: str
    model_identity: Optional[DepthModelIdentity] = Field(None, description="Exact weights and inference preparation used for the fit; null legacy identity requires a new fit")
    scale_s: float = Field(description="Scale factor: metric_depth = raw_depth * scale_s + offset_b")
    offset_b: float = Field(description="Bias/offset in meters")
    residuals: Optional[List[float]] = Field(None, description="Per-point residuals in meters")
    mean_residual_m: Optional[float] = Field(None, description="Mean absolute residual in meters")
    max_residual_m: Optional[float] = Field(None, description="Max absolute residual in meters")
    clip_min_m: float
    clip_max_m: float
    is_active: bool
    created_at: datetime
    
class DepthScaleListResponse(BaseModel):
    items: List[DepthScaleResponse]
    total: int
