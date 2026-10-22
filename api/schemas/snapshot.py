from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class SnapshotResponse(BaseModel):
    id: UUID
    camera_id: UUID
    width: int
    height: int
    size_bytes: int
    captured_at: datetime = Field(description="UTC worker-reception time, not the camera sensor timestamp")
    image_url: str
