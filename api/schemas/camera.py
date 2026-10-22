"""Pydantic schemas for Camera API."""
from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from geoalchemy2.shape import to_shape
from shapely.geometry import Point, Polygon


class CameraBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rtsp_url: Optional[str] = Field(None, max_length=500)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    resolution_w: int = Field(..., gt=0, le=10000)
    resolution_h: int = Field(..., gt=0, le=10000)
    fps: int = Field(25, gt=0, le=120)
    roi_polygon: Optional[List[List[float]]] = None  # [[lon, lat], ...]
    metadata: dict = Field(default_factory=dict)


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    rtsp_url: Optional[str] = Field(None, max_length=500)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    resolution_w: Optional[int] = Field(None, gt=0, le=10000)
    resolution_h: Optional[int] = Field(None, gt=0, le=10000)
    fps: Optional[int] = Field(None, gt=0, le=120)
    roi_polygon: Optional[List[List[float]]] = None
    metadata: Optional[dict] = None


class CameraResponse(CameraBase):
    rtsp_url: Optional[str] = Field(None, deprecated=True, description="Never returned; submit source URLs only on writes.")
    rtsp_configured: bool = False
    stream_enabled: bool = False
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

    @field_serializer("rtsp_url")
    def hide_rtsp_url(self, value: Optional[str]) -> None:
        return None
    
    @classmethod
    def from_orm_with_geo(cls, obj) -> "CameraResponse":
        """Convert from ORM with GeoAlchemy geometry to lat/lon."""
        data = {
            "id": obj.id,
            "name": obj.name,
            "rtsp_url": None,
            "rtsp_configured": bool(obj.rtsp_url),
            "stream_enabled": bool(obj.stream_enabled),
            "resolution_w": obj.resolution_w,
            "resolution_h": obj.resolution_h,
            "fps": obj.fps,
            "metadata": obj.metadata_,
            "created_at": obj.created_at,
            "updated_at": obj.updated_at,
        }
        if obj.location:
            pt = to_shape(obj.location)
            data["longitude"] = pt.x
            data["latitude"] = pt.y
        if obj.roi_polygon:
            poly = to_shape(obj.roi_polygon)
            data["roi_polygon"] = [list(coord) for coord in poly.exterior.coords]
        return cls(**data)


class CameraListResponse(BaseModel):
    items: List[CameraResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class CameraWithActiveCalib(CameraResponse):
    active_calibration_id: Optional[UUID] = None
    active_calibration_version: Optional[int] = None
    active_calibration_rms: Optional[float] = None
