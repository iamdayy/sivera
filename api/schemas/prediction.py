"""Pydantic schemas for Prediction API."""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


PREDICTION_OUTPUT_TYPES = frozenset({
    "mask", "overlay", "instances_json", "depth_raw_vis", "depth_scaled_vis", "depth_masked",
    "height_map", "lifted_points_npz", "volume_heatmap", "volume_result_npz",
    "bev_image", "bev_mask_proyeksi", "bev_overlay",
})


class PredictionJobCreate(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    source_type: str = Field(..., pattern="^(upload|rtsp_frame|latest_rtsp)$")
    source_path: Optional[str] = None
    timestamp: Optional[datetime] = None
    seg_model: Optional[str] = None
    seg_conf: float = Field(0.25, ge=0.0, le=1.0)
    seg_classes: Optional[List[str]] = None
    volume_cell_m: float = Field(0.05, gt=0.0, le=1.0)
    volume_min_pts_per_cell: int = Field(2, ge=1, le=10000)
    return_outputs: Optional[List[str]] = None
    max_frame_age_sec: int = Field(30, ge=1, le=300)


class PredictionJobResponse(BaseModel):
    id: UUID
    camera_id: UUID
    calibration_id: UUID
    ground_model_id: UUID
    depth_scale_id: UUID
    status: str
    source_type: str
    source_path: Optional[str] = Field(description="Private-storage key, not a public URL or filesystem path")
    source_ts: Optional[datetime] = Field(description="Upload caller timestamp or RTSP worker-reception UTC time; not sensor time")
    max_frame_age_sec: int = 30
    volume_cell_m: float = 0.05
    volume_min_pts_per_cell: int = 2
    seg_model: Optional[str]
    seg_conf: float
    seg_classes: Optional[List[str]]
    return_outputs: Optional[List[str]] = None
    result: Optional[dict]
    error_msg: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PredictionJobSubmitResponse(BaseModel):
    job_id: UUID
    status: str
    message: str
    estimated_duration_sec: int = Field(30, description="Fixed advisory estimate, not measured progress or an SLA")


class PredictionJobListResponse(BaseModel):
    items: List[PredictionJobResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class PredictionOutputResponse(BaseModel):
    id: UUID
    job_id: UUID
    output_type: str
    file_path: str = Field(description="Storage key, not a public URL; download through the authenticated job output endpoint")
    mime_type: Optional[str]
    width: Optional[int]
    height: Optional[int]
    meta: Optional[dict]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PredictionOutputsResponse(BaseModel):
    outputs: List[PredictionOutputResponse]


class DetectedInstanceResponse(BaseModel):
    id: UUID
    job_id: UUID
    cls_name: str
    confidence: float
    polygon_px: List[List[float]]
    polygon_world: Optional[List[List[float]]]
    volume_m3: Optional[float]
    footprint_m2: Optional[float]
    height_mean_m: Optional[float]
    height_max_m: Optional[float]
    n_pixels: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class VolumeEstimateResponse(BaseModel):
    id: UUID
    job_id: UUID
    method: str
    volume_m3: float
    volume_liter: float
    footprint_m2: float
    cell_m: Optional[float]
    height_mean_m: Optional[float]
    height_median_m: Optional[float]
    n_cells: Optional[int]
    n_points: Optional[int]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
