from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    version: str
    database: Literal["connected", "disconnected"]


class RootResponse(BaseModel):
    name: str
    version: str
    docs: str
    health: str
