"""Safe prediction error codes shared by runners."""
from api.services.prediction_compute_service import COMPUTE_ERRORS


SAFE_ERRORS = COMPUTE_ERRORS | {
    "prediction_prerequisites_missing", "prediction_source_changed", "prediction_timeout",
    "snapshot_not_found", "snapshot_stale", "snapshot_source_changed", "snapshot_requires_enabled_stream",
    "capture_timeout", "capture_stream_unavailable", "capture_resolution_too_large", "capture_frame_too_large",
    "capture_invalid_frame", "capture_stream_sync_pending", "capture_ffmpeg_missing", "capture_runtime_missing",
    "stream_service_not_configured", "snapshot_image_missing", "snapshot_resolution_mismatch",
    "snapshot_stream_sync_pending", "vision_message_too_large", "vision_runtime_missing",
    "prediction_requires_calibration", "prediction_requires_ground_model", "prediction_requires_depth_scale",
}


def prediction_error(exc: Exception) -> str:
    from api.services.vision_process_service import VisionError
    from api.services.rtsp_capture_service import CaptureError
    if isinstance(exc, TimeoutError) or isinstance(exc, VisionError) and exc.code == "vision_timeout":
        return "prediction_timeout"
    code = exc.code if isinstance(exc, (CaptureError, VisionError)) else str(exc) if isinstance(exc, ValueError) else ""
    return code if code in SAFE_ERRORS else "prediction_processing_failed"
