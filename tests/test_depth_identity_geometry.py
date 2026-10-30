import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from api.core.config import settings
from api.services import depth_service
from api.services.depth_scale_service import validate_saved_scale
from tests.test_depth_scale_routes import setup


@pytest.mark.parametrize("change", ["weights", "preprocessing", "imgsz", "missing"])
def test_activation_readiness_and_submission_reject_changed_identity(setup, change, monkeypatch):
    from api.services.prediction_orchestrator import submit_prediction_job
    response = setup.client.post(setup.url, json=setup.request)
    assert response.status_code == 201, response.text
    scale = setup.rows[depth_service.DepthScale]
    if change == "weights":
        (Path(settings.MODELS_DIR) / scale.depth_model).write_bytes(b"replaced weights")
    elif change == "preprocessing":
        scale.model_identity = dict(scale.model_identity, preprocessing_version="old-v0")
    elif change == "imgsz":
        monkeypatch.setattr(settings, "DEFAULT_DEPTH_IMGSZ", settings.DEFAULT_DEPTH_IMGSZ + 32)
    else:
        scale.model_identity = None
    expected = "depth_scale_identity_missing" if change == "missing" else "depth_scale_model_mismatch"
    for method, suffix in [("get", "/active"), ("post", f"/{scale.id}/activate")]:
        rejected = getattr(setup.client, method)(setup.url + suffix)
        assert rejected.status_code == 409 and rejected.json()["detail"]["code"] == expected
    with pytest.raises(ValueError, match=expected):
        asyncio.run(submit_prediction_job(setup.db, setup.camera.id, "upload"))


def test_changed_weights_during_fit_do_not_replace_active_scale(setup):
    from tests.test_depth_scale_service import inference_result
    async def infer(mode, payload, timeout):
        result = inference_result(payload)
        Path(payload["model_path"]).write_bytes(b"changed during inference")
        return result
    setup.inference.side_effect = infer
    response = setup.client.post(setup.url, json=setup.request)
    assert response.status_code == 409
    assert setup.old.is_active
    setup.db.add.assert_not_called()


def test_saved_fit_rechecks_relative_residuals(setup):
    assert setup.client.post(setup.url, json=setup.request).status_code == 201
    scale = setup.rows[depth_service.DepthScale]
    scale.residuals = [.2, 0, 0]
    scale.mean_residual_m = .2 / 3
    scale.max_residual_m = .2
    scale.fit_provenance["known_distances"][0]["distance_m"] = .1
    with pytest.raises(ValueError, match="depth_scale_invalid_saved_fit"):
        validate_saved_scale(scale)


def test_depth_cache_reloads_same_path_when_weights_change(monkeypatch, tmp_path):
    import ultralytics
    weights = tmp_path / "depth.pt"
    weights.write_bytes(b"first")
    model = SimpleNamespace(task="depth", predict=MagicMock(return_value=[SimpleNamespace(depth=np.ones((3, 4)))]))
    loader = MagicMock(return_value=model)
    monkeypatch.setattr(ultralytics, "YOLO", loader)
    monkeypatch.setattr(depth_service, "_depth_model_cache", {})
    image = np.zeros((3, 4, 3), np.uint8)
    for _ in range(2):
        np.testing.assert_equal(depth_service.predict_depth(image, str(weights)), np.ones((3, 4)))
    assert loader.call_count == 1
    weights.write_bytes(b"other")
    depth_service.predict_depth(image, str(weights))
    assert loader.call_count == 2
    model.predict.side_effect = lambda **kwargs: (weights.write_bytes(b"raced"), [SimpleNamespace(depth=np.ones((3, 4)))])[1]
    with pytest.raises(ValueError, match="depth_scale_model_mismatch"):
        depth_service.predict_depth(image, str(weights))
    assert not depth_service._depth_model_cache


@pytest.mark.parametrize("depth,scale,bias", [([0], 1, 0), ([-1], 1, 5), ([np.nan], 1, 0),
    ([np.inf], 1, 0), ([1], -1, 5), ([1], 1, -2), ([1], 1, np.inf)])
def test_clipping_cannot_hide_invalid_raw_or_corrected_depth(depth, scale, bias):
    with pytest.raises(ValueError, match="prediction_invalid_depth"):
        depth_service.apply_scale(np.array(depth), scale, bias)


def test_positive_depth_clipping_and_camera_axis_lifting():
    np.testing.assert_allclose(depth_service.apply_scale(np.array([.01, 2, 60]), 1, 0), [.1, 2, 50])
    # Optical Z stays constant off-axis; radial distance increases with the ray angle.
    K = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    R = np.diag([1, -1, -1])
    X, Y, H, uv = depth_service.lift_to_ground(K, R, np.array([0, 0, 5]), np.full((2, 3), 4.), np.ones((2, 3), bool))
    np.testing.assert_allclose(X, [0, 2, 4, 0, 2, 4])
    np.testing.assert_allclose(Y, [0, 0, 0, -2, -2, -2])
    np.testing.assert_allclose(H, 1)
    np.testing.assert_array_equal(uv[-1], [2, 1])


def test_invalid_images_never_trigger_ultralytics_plugin_install(monkeypatch):
    from PIL import Image
    from api.services.image_service import decode_prediction_image, encode_image
    from api.services.rtsp_capture_service import decode_frame, CaptureError
    opener = MagicMock(side_effect=AssertionError("global plugin installer must not run"))
    monkeypatch.setattr(Image, "open", opener)
    for extension in (".jpg", ".png", ".bmp"):
        content = encode_image(np.zeros((8, 8, 3), np.uint8), extension)
        assert decode_prediction_image(content).shape == (8, 8, 3)
        if extension == ".jpg":
            assert decode_frame(content).width == 8
    for content in (b"invalid", b"\xff\xd8\xffbroken", b"\x89PNG\r\n\x1a\nbroken", b"BMbroken"):
        with pytest.raises(ValueError, match="prediction_invalid_image"):
            decode_prediction_image(content)
        with pytest.raises(CaptureError, match="capture_invalid_frame"):
            decode_frame(content)
    opener.assert_not_called()
