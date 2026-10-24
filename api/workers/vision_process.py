import base64
import json
import sys
from dataclasses import asdict

from api.core.config import Settings
from api.services.calibration_quality_service import assess_frame
from api.services.calibration_service import calibrate_from_points
from api.services.rtsp_capture_service import decode_frame


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        if sys.argv[1] == "assess":
            config = Settings(_env_file=None, CALIB_CAPTURE_MIN_SHARPNESS=payload["min_sharpness"],
                              CAPTURE_MAX_FRAME_BYTES=payload["max_frame_bytes"], CAPTURE_MAX_PIXELS=payload["max_pixels"])
            frame = decode_frame(base64.b64decode(payload["image"]), config)
            result = asdict(assess_frame(frame, payload["cols"], payload["rows"], payload["previous"], config))
        elif sys.argv[1] == "calibrate":
            result = calibrate_from_points(payload["points"], tuple(payload["size"]), payload["cols"], payload["rows"], payload["square_m"])
        elif sys.argv[1] == "depth-scale":
            import contextlib
            import cv2
            from api.schemas.ground_ref import DepthScaleCreate
            from api.services.depth_service import predict_depth
            from api.services.depth_scale_service import fit_depth_scale
            config = Settings(_env_file=None, DEPTH_SCALE_MAX_MEAN_RESIDUAL_M=payload["max_mean"],
                              DEPTH_SCALE_MAX_RESIDUAL_M=payload["max_residual"],
                              DEPTH_SCALE_MAX_RELATIVE_RESIDUAL=payload["max_relative"])
            # Model libraries may print to stdout; reserve stdout for the JSON protocol.
            with contextlib.redirect_stdout(sys.stderr):
                depth = predict_depth(cv2.imread(payload["image_path"]), payload["model_path"], payload["imgsz"], "cpu")
            result = fit_depth_scale(depth, DepthScaleCreate.model_validate(payload["body"]), config)
        else:
            raise ValueError("vision_failed")
    except ValueError as exc:
        code = str(exc)
        result = {"error": code if code in {"calibration_insufficient_images", "calibration_invalid_corners", "calibration_invalid_result", "calibration_invalid_parameters", "depth_scale_invalid_depth", "depth_scale_pixel_out_of_bounds", "depth_scale_degenerate_fit", "depth_scale_poor_fit"} else "vision_failed"}
    except Exception:
        result = {"error": "vision_failed"}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
