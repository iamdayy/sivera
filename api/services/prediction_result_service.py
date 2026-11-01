import asyncio
import base64
import io
import logging
import math
from collections.abc import Awaitable
from typing import TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.storage import StorageBackend, generate_storage_key
from api.models.orm import DetectedInstance, PredictionJob, PredictionOutput, VolumeEstimate
from api.schemas.prediction import PREDICTION_OUTPUT_TYPES


logger = logging.getLogger(__name__)
T = TypeVar("T")


async def finish_io(work: Awaitable[T]) -> T:
    # A cancelled storage thread must finish before attempt-owned files are removed.
    task = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


def grid_volume(result: dict) -> float | None:
    row = next((item for item in result["volumes"] if item["method"] == "grid"), None)
    value = row.get("volume_liter") if row else None
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


async def last_saved_volume(db: AsyncSession, camera_id: UUID) -> float | None:
    return await db.scalar(
        select(VolumeEstimate.volume_liter).join(PredictionJob, VolumeEstimate.job_id == PredictionJob.id)
        .where(PredictionJob.camera_id == camera_id, PredictionJob.status == "done", VolumeEstimate.method == "grid")
        .order_by(PredictionJob.completed_at.desc(), PredictionJob.id.desc(), VolumeEstimate.id.desc()).limit(1)
    )


class PredictionArtifacts:
    def __init__(self, storage: StorageBackend, camera_id: UUID, job_id: UUID) -> None:
        self.storage, self.camera_id, self.job_id = storage, camera_id, job_id
        self.outputs: list[dict] = []
        self.keys: list[str] = []
        self.committed = False

    async def write(self, artifacts: dict, selected: list[str] | None) -> None:
        allowed = PREDICTION_OUTPUT_TYPES if selected is None else set(selected)
        for kind, artifact in artifacts.items():
            if kind not in allowed or kind not in PREDICTION_OUTPUT_TYPES:
                continue
            key = generate_storage_key(str(self.camera_id), str(self.job_id), kind, artifact["extension"])
            self.keys.append(key)
            await finish_io(asyncio.to_thread(
                self.storage.save, key, io.BytesIO(base64.b64decode(artifact["data"], validate=True)), artifact["mime_type"],
            ))
            self.outputs.append({"output_type": kind, "file_path": key,
                                 **{name: artifact.get(name) for name in ("mime_type", "width", "height", "meta")}})

    def add_rows(self, db: AsyncSession, result: dict) -> None:
        for output in self.outputs:
            db.add(PredictionOutput(job_id=self.job_id, **output))
        for instance in result["instances"]:
            db.add(DetectedInstance(job_id=self.job_id, **instance))
        for volume in result["volumes"]:
            db.add(VolumeEstimate(job_id=self.job_id, **volume))

    async def cleanup(self) -> None:
        if self.committed:
            return
        for key in self.keys:
            try:
                await finish_io(asyncio.to_thread(self.storage.delete, key))
            except Exception:
                logger.warning("Prediction artifact cleanup failed job=%s", self.job_id)
