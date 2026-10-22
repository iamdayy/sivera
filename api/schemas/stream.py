from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StreamUpdate(BaseModel):
    enabled: bool
    model_config = ConfigDict(extra="forbid")


class StreamResponse(BaseModel):
    camera_id: UUID
    enabled: bool
    status: Literal["disabled", "idle", "online", "offline", "unknown"]
    synced: bool
    whep_url: str | None = None
    tracks: list[str] = Field(default_factory=list)
    checked_at: datetime
