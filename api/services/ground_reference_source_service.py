import asyncio
import hashlib
import hmac
import io
from uuid import UUID, uuid4

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.storage import StorageBackend
from api.models.orm import Calibration, Camera, CameraSnapshot, GroundReferenceSource
from api.services.image_service import decode_prediction_image, encode_image
from api.services.rtsp_capture_service import CaptureError
from api.services.snapshot_service import require_rtsp_camera


PREPARATION_VERSION = f"undistorted-pixels-v1-opencv-{cv2.__version__}"


class GroundReferenceError(Exception):
    def __init__(self, code: str, status: int = 409):
        self.code = code
        self.status = status
        super().__init__(code)


async def lock_camera(db: AsyncSession, camera_id: UUID) -> Camera:
    camera = (await db.execute(select(Camera).where(Camera.id == camera_id)
        .with_for_update().execution_options(populate_existing=True))).scalar_one_or_none()
    if camera is None:
        raise GroundReferenceError("camera_not_found", 404)
    return camera


async def active_calibration(db: AsyncSession, camera_id: UUID, expected_id: UUID | None = None) -> Calibration:
    calibration = (await db.execute(select(Calibration).where(
        Calibration.camera_id == camera_id, Calibration.is_active == True)
        .order_by(Calibration.version.desc()).limit(1)
        .execution_options(populate_existing=True))).scalar_one_or_none()
    if calibration is None:
        raise GroundReferenceError("ground_reference_requires_calibration")
    if expected_id is not None and calibration.id != expected_id:
        raise GroundReferenceError("ground_reference_calibration_changed")
    return calibration


def require_source_context(camera: Camera, calibration: Calibration, source: CameraSnapshot | GroundReferenceSource) -> None:
    if camera.id != source.camera_id or calibration.camera_id != camera.id:
        raise GroundReferenceError("ground_reference_source_not_found", 404)
    try:
        current_hash = require_rtsp_camera(camera)
    except CaptureError as exc:
        raise GroundReferenceError(exc.code) from None
    if not hmac.compare_digest(source.source_hash, current_hash):
        raise GroundReferenceError("ground_reference_source_changed")
    if calibration.k_original is None or not calibration.resolution_w or not calibration.resolution_h:
        raise GroundReferenceError("calibration_intrinsics_missing")
    if ((source.width, source.height) != (camera.resolution_w, camera.resolution_h)
            or (source.width, source.height) != (calibration.resolution_w, calibration.resolution_h)):
        raise GroundReferenceError("ground_reference_resolution_changed")


async def require_source(db: AsyncSession, camera_id: UUID, source_id: UUID) -> GroundReferenceSource:
    source = (await db.execute(select(GroundReferenceSource).where(
        GroundReferenceSource.id == source_id, GroundReferenceSource.camera_id == camera_id
    ))).scalar_one_or_none()
    if source is None:
        raise GroundReferenceError("ground_reference_source_not_found", 404)
    return source


def read_source_bytes(source: GroundReferenceSource, storage: StorageBackend) -> bytes:
    limit = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if not 0 < source.size_bytes <= limit:
        raise GroundReferenceError("ground_reference_source_image_changed")
    stream = storage.get(source.storage_key)
    if stream is None:
        raise GroundReferenceError("ground_reference_source_image_missing", 404)
    try:
        content = stream.read(source.size_bytes + 1)
    finally:
        stream.close()
    if len(content) != source.size_bytes or not hmac.compare_digest(hashlib.sha256(content).hexdigest(), source.image_sha256):
        raise GroundReferenceError("ground_reference_source_image_changed")
    return content


def read_source_image(source: GroundReferenceSource, storage: StorageBackend) -> np.ndarray:
    image = decode_prediction_image(read_source_bytes(source, storage))
    if image.shape[1::-1] != (source.width, source.height):
        raise GroundReferenceError("ground_reference_source_image_changed")
    return image


def prepare_snapshot(snapshot: CameraSnapshot, calibration: Calibration, camera: Camera, storage: StorageBackend) -> bytes:
    from api.services.prediction_orchestrator import prepare_prediction_image

    stream = storage.get(snapshot.storage_key)
    if stream is None:
        raise GroundReferenceError("snapshot_not_found", 404)
    try:
        content = stream.read(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)
    finally:
        stream.close()
    image = decode_prediction_image(content)
    if image.shape[1::-1] != (snapshot.width, snapshot.height):
        raise GroundReferenceError("ground_reference_resolution_changed")
    prepared = prepare_prediction_image(image, calibration, camera, "rtsp_frame")
    content = encode_image(prepared, ".png")
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise ValueError("prediction_image_too_large")
    return content


async def prepare_source(db: AsyncSession, camera_id: UUID, snapshot_id: UUID,
                         calibration_id: UUID, storage: StorageBackend) -> tuple[GroundReferenceSource, bool]:
    camera = await lock_camera(db, camera_id)
    calibration = await active_calibration(db, camera_id, calibration_id)
    snapshot = (await db.execute(select(CameraSnapshot).where(
        CameraSnapshot.id == snapshot_id, CameraSnapshot.camera_id == camera_id))).scalar_one_or_none()
    if snapshot is None:
        raise GroundReferenceError("snapshot_not_found", 404)
    require_source_context(camera, calibration, snapshot)
    existing = (await db.execute(select(GroundReferenceSource).where(
        GroundReferenceSource.snapshot_id == snapshot_id, GroundReferenceSource.calibration_id == calibration_id,
        GroundReferenceSource.preparation_version == PREPARATION_VERSION))).scalar_one_or_none()
    if existing is not None:
        try:
            await asyncio.to_thread(read_source_bytes, existing, storage)
            return existing, False
        except GroundReferenceError as exc:
            if exc.code != "ground_reference_source_image_missing":
                raise

    content = await asyncio.to_thread(prepare_snapshot, snapshot, calibration, camera, storage)
    digest = hashlib.sha256(content).hexdigest()
    if existing is not None:
        if digest != existing.image_sha256 or len(content) != existing.size_bytes:
            raise GroundReferenceError("ground_reference_source_image_changed")
        source = existing
    else:
        source_id = uuid4()
        source = GroundReferenceSource(id=source_id, camera_id=camera_id, snapshot_id=snapshot_id,
            calibration_id=calibration_id, source_hash=snapshot.source_hash, image_sha256=digest,
            preparation_version=PREPARATION_VERSION, width=snapshot.width, height=snapshot.height,
            size_bytes=len(content), captured_at=snapshot.captured_at,
            storage_key=f"cameras/{camera_id}/ground-reference-sources/{source_id}/undistorted.png")
    await asyncio.to_thread(storage.save, source.storage_key, io.BytesIO(content), "image/png")
    if existing is None:
        db.add(source)
    await db.flush()
    return source, existing is None
