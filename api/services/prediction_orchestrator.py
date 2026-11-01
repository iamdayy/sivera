"""Prediction prerequisites, pinned context and image-coordinate contract."""
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import raiseload

from api.core.config import settings
from api.models.orm import Camera, Calibration, GroundModel, GroundReference, DepthScale, PredictionJob
from api.services.depth_model_service import validate_scale_model
from api.services.depth_scale_service import validate_saved_scale


def prediction_options(job: PredictionJob) -> dict:
    return {name: getattr(job, name) for name in (
        "source_type", "seg_model", "seg_conf", "seg_classes", "volume_cell_m",
        "volume_min_pts_per_cell", "return_outputs", "max_frame_age_sec",
    )}


def measurement_context(camera: Camera, calibration: Calibration, ground: GroundModel, scale: DepthScale) -> dict:
    def fields(row, names):
        return {name: getattr(row, name) for name in names.split()}

    return {
        "ids": {"calibration_id": str(calibration.id), "ground_model_id": str(ground.id), "depth_scale_id": str(scale.id)},
        "camera": fields(camera, "resolution_w resolution_h"),
        "calibration": fields(calibration, "k_matrix k_original k_undist dist_coeffs resolution_w resolution_h"),
        "ground_model": fields(ground, "R_matrix t_vector H_i2w H_w2i cam_height pitch_down_deg bev_ppm bev_margin_m"),
        "depth_scale": fields(scale, "depth_model model_identity fit_provenance scale_s offset_b residuals mean_residual_m max_residual_m clip_min_m clip_max_m"),
    }


async def active_prerequisites(db: AsyncSession, camera_id: UUID, source_type: str) -> tuple[Calibration, GroundModel, DepthScale]:
    calibration = (await db.execute(
        select(Calibration).options(raiseload("*")).where(Calibration.camera_id == camera_id, Calibration.is_active.is_(True))
        .order_by(Calibration.version.desc()).limit(1)
    )).scalar_one_or_none()
    if calibration is None:
        raise ValueError("prediction_requires_calibration")
    if source_type != "upload" and (calibration.k_original is None or not calibration.resolution_w or not calibration.resolution_h):
        raise ValueError("calibration_intrinsics_missing")
    ground = (await db.execute(
        select(GroundModel).options(raiseload("*")).join(GroundModel.ground_reference)
        .where(GroundModel.camera_id == camera_id, GroundModel.calibration_id == calibration.id,
               GroundReference.camera_id == camera_id, GroundReference.calibration_id == calibration.id,
               GroundReference.is_active.is_(True))
        .order_by(GroundReference.version.desc(), GroundModel.created_at.desc(), GroundModel.id.desc()).limit(1)
    )).scalar_one_or_none()
    if ground is None:
        raise ValueError("prediction_requires_ground_model")
    scale = (await db.execute(
        select(DepthScale).options(raiseload("*")).where(
            DepthScale.camera_id == camera_id, DepthScale.ground_model_id == ground.id, DepthScale.is_active.is_(True),
        ).order_by(DepthScale.created_at.desc(), DepthScale.id.desc()).limit(1)
    )).scalar_one_or_none()
    if scale is None:
        raise ValueError("prediction_requires_depth_scale")
    return calibration, ground, scale


async def pinned_context(db: AsyncSession, job: PredictionJob) -> dict:
    calibration = await db.get(Calibration, job.calibration_id, options=[raiseload("*")])
    ground = await db.get(GroundModel, job.ground_model_id, options=[raiseload("*")])
    scale = await db.get(DepthScale, job.depth_scale_id, options=[raiseload("*")])
    camera = await db.get(Camera, job.camera_id, options=[raiseload("*")])
    if (camera is None or calibration is None or ground is None or scale is None
            or calibration.camera_id != job.camera_id or ground.camera_id != job.camera_id
            or ground.calibration_id != calibration.id or scale.camera_id != job.camera_id
            or scale.ground_model_id != ground.id):
        raise ValueError("prediction_prerequisites_missing")
    reference = await db.get(GroundReference, ground.ground_ref_id, options=[raiseload("*")])
    if reference is None or reference.camera_id != job.camera_id or reference.calibration_id != calibration.id:
        raise ValueError("prediction_prerequisites_missing")
    return measurement_context(camera, calibration, ground, scale)


def prepare_prediction_image(image: np.ndarray, calibration: Calibration, camera: Camera, source_type: str) -> np.ndarray:
    if image is None:
        raise ValueError("prediction_invalid_image")
    size = (calibration.resolution_w, calibration.resolution_h)
    if calibration.k_original is not None and not all(size):
        raise ValueError("calibration_invalid_intrinsics")
    explicit = calibration.k_original is not None and all(size)
    if source_type != "upload" and not explicit:
        raise ValueError("calibration_intrinsics_missing")
    expected = size if explicit else (camera.resolution_w, camera.resolution_h)
    if image.shape[1::-1] != expected:
        raise ValueError("prediction_resolution_mismatch")
    registered = np.array(calibration.k_matrix, np.float64)
    if (registered.shape != (3, 3) or not np.isfinite(registered).all()
            or min(registered[0, 0], registered[1, 1]) <= 0
            or not np.allclose(registered[2], [0, 0, 1])):
        raise ValueError("calibration_invalid_intrinsics")
    if not explicit:
        # Legacy uploads retain the historical already-undistorted input contract.
        return image
    original = np.array(calibration.k_original, np.float64)
    target = np.array(calibration.k_undist or calibration.k_matrix, np.float64)
    distortion = np.array(calibration.dist_coeffs, np.float64).ravel()
    if (original.shape != (3, 3) or target.shape != (3, 3) or len(distortion) not in {4, 5, 8, 12, 14}
            or not np.isfinite(original).all() or not np.isfinite(target).all() or not np.isfinite(distortion).all()
            or min(original[0, 0], original[1, 1], target[0, 0], target[1, 1]) <= 0
            or not np.allclose(original[2], [0, 0, 1]) or not np.allclose(target, registered)):
        raise ValueError("calibration_invalid_intrinsics")
    return cv2.undistort(image, original, distortion, None, target)


async def submit_prediction_job(
    db: AsyncSession, camera_id: UUID, source_type: str,
    source_path: str | None = None, source_ts: datetime | None = None,
    seg_model: str | None = None, seg_conf: float = 0.25, seg_classes: list[str] | None = None,
    volume_cell_m: float = 0.05, volume_min_pts_per_cell: int = 2,
    return_outputs: list[str] | None = None, max_frame_age_sec: int = 30, source_hash: str | None = None,
) -> PredictionJob:
    calibration, ground, scale = await active_prerequisites(db, camera_id, source_type)
    await asyncio.to_thread(validate_scale_model, scale)
    validate_saved_scale(scale)
    now = datetime.now(timezone.utc)
    job = PredictionJob(
        camera_id=camera_id, calibration_id=calibration.id, ground_model_id=ground.id, depth_scale_id=scale.id,
        status="pending", source_type=source_type, source_path=source_path, source_ts=source_ts,
        seg_model=seg_model, seg_conf=seg_conf, seg_classes=seg_classes, return_outputs=return_outputs,
        volume_cell_m=volume_cell_m, volume_min_pts_per_cell=volume_min_pts_per_cell,
        max_frame_age_sec=max_frame_age_sec, source_hash=source_hash,
        created_at=now, deadline_at=now + timedelta(seconds=settings.CAPTURE_QUEUE_TIMEOUT_SEC),
    )
    db.add(job)
    await db.flush()
    return job
