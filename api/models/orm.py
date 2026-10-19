"""SQLAlchemy ORM models for the Waste Volume Estimation API."""
import enum
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from geoalchemy2 import Geography, Geometry
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.core.database import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, enum.Enum):
    UPLOAD = "upload"
    RTSP_FRAME = "rtsp_frame"
    LATEST_RTSP = "latest_rtsp"


class OutputType(str, enum.Enum):
    MASK = "mask"
    OVERLAY = "overlay"
    BEV_IMAGE = "bev_image"
    BEV_MASK_PROYEKSI = "bev_mask_proyeksi"
    BEV_OVERLAY = "bev_overlay"
    DEPTH_RAW_VIS = "depth_raw_vis"
    DEPTH_SCALED_VIS = "depth_scaled_vis"
    DEPTH_MASKED = "depth_masked"
    HEIGHT_MAP = "height_map"
    VOLUME_HEATMAP = "volume_heatmap"
    NPZ_LIFTED_POINTS = "lifted_points_npz"
    NPZ_VOLUME_RESULT = "volume_result_npz"
    NPZ_DEPTH_SCALE = "depth_scale_npz"
    INSTANCES_JSON = "instances_json"


class Camera(Base):
    __tablename__ = "cameras"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    rtsp_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    location: Mapped[Optional[Geography]] = mapped_column(
        Geography(geometry_type="POINT", srid=4326), nullable=True
    )
    resolution_w: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution_h: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[int] = mapped_column(Integer, default=25)
    roi_polygon: Mapped[Optional[Geography]] = mapped_column(
        Geography(geometry_type="POLYGON", srid=4326), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    
    # Relationships
    calibrations: Mapped[List["Calibration"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", lazy="selectin"
    )
    ground_references: Mapped[List["GroundReference"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", lazy="selectin"
    )
    ground_models: Mapped[List["GroundModel"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", lazy="selectin"
    )
    depth_scales: Mapped[List["DepthScale"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", lazy="selectin"
    )
    prediction_jobs: Mapped[List["PredictionJob"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", lazy="selectin"
    )
    api_keys: Mapped[List["ApiKey"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan", lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, name='{self.name}', resolution={self.resolution_w}x{self.resolution_h})>"


class Calibration(Base):
    __tablename__ = "calibrations"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    k_matrix: Mapped[list] = mapped_column(JSON, nullable=False)  # 3x3 list
    dist_coeffs: Mapped[list] = mapped_column(JSON, nullable=False)  # [k1,k2,p1,p2,k3]
    rms_reproj: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coverage_w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coverage_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    chessboard_cols: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chessboard_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    square_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calib_images: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="calibrations")
    ground_references: Mapped[List["GroundReference"]] = relationship(
        back_populates="calibration", cascade="all, delete-orphan", lazy="selectin"
    )
    ground_models: Mapped[List["GroundModel"]] = relationship(
        back_populates="calibration", cascade="all, delete-orphan", lazy="selectin"
    )
    
    __table_args__ = (
        UniqueConstraint("camera_id", "version", name="uq_calibration_camera_version"),
        Index("ix_calibrations_camera_active", "camera_id", "is_active"),
    )
    
    def __repr__(self) -> str:
        return f"<Calibration(id={self.id}, camera_id={self.camera_id}, v={self.version}, active={self.is_active})>"


class GroundReference(Base):
    __tablename__ = "ground_references"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    calibration_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calibrations.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    image_points: Mapped[list] = mapped_column(JSON, nullable=False)  # [[u,v],...]
    world_points: Mapped[list] = mapped_column(JSON, nullable=False)  # [[X,Y],...] meters
    world_units: Mapped[str] = mapped_column(String(10), default="meter", nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="ground_references")
    calibration: Mapped["Calibration"] = relationship(back_populates="ground_references")
    ground_models: Mapped[List["GroundModel"]] = relationship(
        back_populates="ground_reference", cascade="all, delete-orphan", lazy="selectin"
    )
    
    __table_args__ = (
        UniqueConstraint("camera_id", "version", name="uq_ground_ref_camera_version"),
        Index("ix_ground_refs_camera_active", "camera_id", "is_active"),
    )
    
    def __repr__(self) -> str:
        return f"<GroundReference(id={self.id}, camera_id={self.camera_id}, v={self.version}, n_pts={len(self.image_points)})>"


class GroundModel(Base):
    __tablename__ = "ground_models"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    calibration_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calibrations.id", ondelete="CASCADE"), nullable=False
    )
    ground_ref_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ground_references.id", ondelete="CASCADE"), nullable=False
    )
    R_matrix: Mapped[list] = mapped_column(JSON, nullable=False)  # 3x3
    t_vector: Mapped[list] = mapped_column(JSON, nullable=False)  # 3x1
    H_i2w: Mapped[list] = mapped_column(JSON, nullable=False)  # 3x3
    H_w2i: Mapped[list] = mapped_column(JSON, nullable=False)  # 3x3
    cam_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pitch_down_deg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reproj_err_px: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bev_ppm: Mapped[int] = mapped_column(Integer, default=200, nullable=False)
    bev_margin_m: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="ground_models")
    calibration: Mapped["Calibration"] = relationship(back_populates="ground_models")
    ground_reference: Mapped["GroundReference"] = relationship(back_populates="ground_models")
    depth_scales: Mapped[List["DepthScale"]] = relationship(
        back_populates="ground_model", cascade="all, delete-orphan", lazy="selectin"
    )
    prediction_jobs: Mapped[List["PredictionJob"]] = relationship(
        back_populates="ground_model", lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<GroundModel(id={self.id}, cam_h={self.cam_height}, pitch={self.pitch_down_deg})>"


class DepthScale(Base):
    __tablename__ = "depth_scales"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ground_model_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ground_models.id", ondelete="CASCADE"), nullable=False
    )
    depth_model: Mapped[str] = mapped_column(String(100), nullable=False)
    scale_s: Mapped[float] = mapped_column(Float, nullable=False)
    offset_b: Mapped[float] = mapped_column(Float, nullable=False)
    residuals: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    mean_residual_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_residual_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clip_min_m: Mapped[float] = mapped_column(Float, default=0.1, nullable=False)
    clip_max_m: Mapped[float] = mapped_column(Float, default=50.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="depth_scales")
    ground_model: Mapped["GroundModel"] = relationship(back_populates="depth_scales")
    prediction_jobs: Mapped[List["PredictionJob"]] = relationship(
        back_populates="depth_scale", lazy="selectin"
    )
    
    def __repr__(self) -> str:
        return f"<DepthScale(id={self.id}, s={self.scale_s:.4f}, b={self.offset_b:.4f})>"


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    calibration_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("calibrations.id", ondelete="CASCADE"), nullable=False
    )
    ground_model_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ground_models.id", ondelete="CASCADE"), nullable=False
    )
    depth_scale_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("depth_scales.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        String(20), default=JobStatus.PENDING, nullable=False, index=True
    )
    source_type: Mapped[SourceType] = mapped_column(String(20), nullable=False)
    source_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source_ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    seg_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    seg_conf: Mapped[float] = mapped_column(Float, default=0.25, nullable=False)
    seg_classes: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_msg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="prediction_jobs")
    calibration: Mapped["Calibration"] = relationship()
    ground_model: Mapped["GroundModel"] = relationship(back_populates="prediction_jobs")
    depth_scale: Mapped["DepthScale"] = relationship(back_populates="prediction_jobs")
    outputs: Mapped[List["PredictionOutput"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )
    instances: Mapped[List["DetectedInstance"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )
    volume_estimates: Mapped[List["VolumeEstimate"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )
    
    __table_args__ = (
        Index("ix_jobs_camera_status", "camera_id", "status"),
        Index("ix_jobs_created", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<PredictionJob(id={self.id}, camera_id={self.camera_id}, status={self.status})>"


class PredictionOutput(Base):
    __tablename__ = "prediction_outputs"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("prediction_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    output_type: Mapped[OutputType] = mapped_column(String(50), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    job: Mapped["PredictionJob"] = relationship(back_populates="outputs")
    
    def __repr__(self) -> str:
        return f"<PredictionOutput(job_id={self.job_id}, type={self.output_type})>"


class DetectedInstance(Base):
    __tablename__ = "detected_instances"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("prediction_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cls_name: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    polygon_px: Mapped[list] = mapped_column(JSON, nullable=False)  # [[u,v],...]
    polygon_world: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [[X,Y],...]
    volume_m3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    footprint_m2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    height_mean_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    height_max_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_pixels: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    job: Mapped["PredictionJob"] = relationship(back_populates="instances")
    
    def __repr__(self) -> str:
        return f"<DetectedInstance(job_id={self.job_id}, cls={self.cls_name}, conf={self.confidence:.2f})>"


class VolumeEstimate(Base):
    __tablename__ = "volume_estimates"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[uuid4] = mapped_column(
        UUID(as_uuid=True), ForeignKey("prediction_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    method: Mapped[str] = mapped_column(String(30), nullable=False)  # "grid", "convex_hull"
    volume_m3: Mapped[float] = mapped_column(Float, nullable=False)
    volume_liter: Mapped[float] = mapped_column(Float, nullable=False)
    footprint_m2: Mapped[float] = mapped_column(Float, nullable=False)
    cell_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    height_mean_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    height_median_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_cells: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_points: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    job: Mapped["PredictionJob"] = relationship(back_populates="volume_estimates")
    
    def __repr__(self) -> str:
        return f"<VolumeEstimate(job_id={self.job_id}, method={self.method}, vol={self.volume_m3:.3f}m3)>"


class ApiKey(Base):
    __tablename__ = "api_keys"
    
    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[Optional[uuid4]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True
    )
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    scopes: Mapped[list] = mapped_column(JSON, default=["predict", "read"], nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    camera: Mapped[Optional["Camera"]] = relationship(back_populates="api_keys")
    
    def __repr__(self) -> str:
        return f"<ApiKey(id={self.id}, name='{self.name}', active={self.is_active})>"