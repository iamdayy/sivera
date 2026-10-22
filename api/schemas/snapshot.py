from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class SnapshotResponse(BaseModel):
    id: UUID
    camera_id: UUID
    width: int
    height: int
    size_bytes: int
    captured_at: datetime
    image_url: str
