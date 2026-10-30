import hashlib
from importlib.metadata import version
from pathlib import Path

from api.core.config import settings
from api.models.orm import DepthScale


def weights_digest(path: Path) -> str:
    try:
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
    except OSError:
        raise ValueError("depth_model_unavailable") from None
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise ValueError("depth_scale_model_mismatch")
    return digest


def depth_model_identity(path: Path, imgsz: int | None = None) -> dict:
    return {
        "sha256": weights_digest(path),
        "ultralytics_version": version("ultralytics"),
        "torch_version": version("torch"),
        "opencv_version": version("opencv-python-headless"),
        "preprocessing_version": "undistorted-bgr-ultralytics-stretch-v1",
        "depth_convention": "camera_axis_z",
        "imgsz": settings.DEFAULT_DEPTH_IMGSZ if imgsz is None else imgsz,
    }


def validate_scale_model(scale: DepthScale) -> Path:
    from api.services.depth_service import resolve_depth_model

    if not getattr(scale, "model_identity", None):
        raise ValueError("depth_scale_identity_missing")
    path = resolve_depth_model(scale.depth_model)
    if scale.model_identity != depth_model_identity(path):
        raise ValueError("depth_scale_model_mismatch")
    return path
