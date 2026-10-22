import asyncio

import cv2
import numpy as np
import pytest

from api.core.config import Settings
from api.services.calibration_quality_service import assess_frame, assess_frame_bounded, corner_preview, similar_pose
from api.services.calibration_service import calibrate_from_points
from api.services.rtsp_capture_service import decode_frame
from api.services.vision_process_service import VisionError, run_vision
from tests.calibration_fixtures import chessboard, projected_corners
from tests.test_calibration_capture_service import jpeg


def test_real_chessboard_corners_sharpness_and_diversity():
    frame = decode_frame(chessboard())
    quality = assess_frame(frame, 9, 6, [])
    assert quality.status == "accepted"
    assert len(quality.corners) == 54
    assert quality.sharpness > 80
    duplicate = assess_frame(frame, 9, 6, [quality.descriptor])
    assert duplicate.reason == "similar_pose"
    moved = assess_frame(decode_frame(chessboard(x=200)), 9, 6, [quality.descriptor])
    assert moved.status == "accepted"
    smaller = assess_frame(decode_frame(chessboard(square=25)), 9, 6, [quality.descriptor])
    assert smaller.status == "accepted"
    preview = corner_preview(frame.content, quality.corners, 9, 6)
    assert (decode_frame(preview).width, decode_frame(preview).height) == (640, 480)
    assert preview != frame.content


def test_blur_and_chessboard_absence():
    assert assess_frame(decode_frame(jpeg()), 9, 6, []).reason == "chessboard_not_found"
    quality = assess_frame(decode_frame(chessboard(blur=4)), 9, 6, [])
    assert quality.reason == "blur"
    assert quality.corners is not None


def test_angle_wrap_and_perspective_diversity():
    a = [0.5, 0.5, 0.4, 0.3, 0.02, 1, 1]
    b = [0.5, 0.5, 0.4, 0.3, np.pi - 0.02, 1, 1]
    assert similar_pose(a, b)
    b[5] = 0.7
    assert not similar_pose(a, b)


def test_calibration_recovers_known_intrinsics_and_units():
    K, points = projected_corners()
    result = calibrate_from_points(points, (640, 480), 9, 6, 0.025)
    np.testing.assert_allclose(result["K"], K, atol=0.1)
    assert result["rms"] < 0.001
    assert result["size"] == [640, 480]
    assert len(result["per_view"]) == 12
    assert all(np.isfinite(result["coverage"]))
    with pytest.raises(ValueError, match="calibration_insufficient_images"):
        calibrate_from_points(points[:2], (640, 480), 9, 6, 0.025)
    points[0][0][0] = float("nan")
    with pytest.raises(ValueError, match="calibration_invalid_corners"):
        calibrate_from_points(points, (640, 480), 9, 6, 0.025)


def test_real_bounded_process_and_timeout():
    quality = asyncio.run(assess_frame_bounded(decode_frame(chessboard()), 9, 6, [], Settings(_env_file=None)))
    assert quality.status == "accepted"
    with pytest.raises(VisionError, match="vision_timeout"):
        asyncio.run(run_vision("calibrate", {}, 0.001))


def test_bounded_process_cancellation():
    async def stopped():
        return False
    with pytest.raises(VisionError, match="vision_cancelled"):
        asyncio.run(run_vision("assess", {}, 15, stopped))


def test_two_corner_axes_rejected_before_opencv():
    from api.schemas.calibration_capture import CaptureCreate
    from api.services.calibration_quality_service import detect_chessboard
    with pytest.raises(ValueError):
        CaptureCreate(chessboard_rows=2)
    with pytest.raises(ValueError, match="calibration_invalid_parameters"):
        detect_chessboard(np.zeros((240, 320), np.uint8), 4, 2)
