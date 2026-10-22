from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.security import check_camera_access
from api.models.orm import ApiKey, CameraSnapshot, PredictionJob, PredictionOutput


async def require_job(db: AsyncSession, camera_id: UUID, job_id: UUID, api_key: ApiKey) -> None:
    check_camera_access(api_key, camera_id)
    found = await db.scalar(select(PredictionJob.id).where(PredictionJob.id == job_id, PredictionJob.camera_id == camera_id))
    if found is None:
        raise HTTPException(status_code=404, detail={"code": "prediction_not_found"})


async def authorize_artifact(db: AsyncSession, key: str, api_key: ApiKey) -> None:
    parts = key.split("/")
    if "\\" in key or any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(status_code=404, detail="Artifact not found")
    if len(parts) < 3 or parts[0] != "cameras":
        if api_key.camera_id is not None:
            raise HTTPException(status_code=403, detail="API key cannot access this artifact")
        return
    try:
        camera_id = UUID(parts[1])
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found") from None
    check_camera_access(api_key, camera_id)
    if "snapshots" in parts:
        found = await db.scalar(select(CameraSnapshot.id).where(CameraSnapshot.storage_key == key, CameraSnapshot.camera_id == camera_id))
        scopes = ("camera:read",)
        if found is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
    elif parts[2] in {"calibrations", "calib_images", "calib_preview"}:
        scopes = ("calibration:read",)
    elif len(parts) == 4 and parts[2] in {"bev_image", "bev_mask", "bev_overlay", "ground_refs", "depth_scale_npz"}:
        scopes = ("ground_ref:read",)
    elif parts[2] in {"uploads", "mask", "overlay", "instances_json", "depth_raw_vis", "depth_scaled_vis", "depth_masked", "height_map", "lifted_points_npz", "volume_heatmap", "volume_result_npz", "bev_image", "bev_mask_proyeksi", "bev_overlay"} or parts[2].startswith("mask_"):
        # Prediction outputs not registered in the DB (e.g. orphan writes) stay private.
        found = await db.scalar(select(PredictionJob.id).outerjoin(PredictionOutput, PredictionOutput.job_id == PredictionJob.id).where(
            PredictionJob.camera_id == camera_id,
            or_(PredictionJob.source_path == key, PredictionOutput.file_path == key),
        ).limit(1))
        if found is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        scopes = ("predict:read",)
    else:
        return
    if "*" not in api_key.scopes and not any(scope in api_key.scopes for scope in scopes):
        raise HTTPException(status_code=403, detail="Artifact read scope required")
