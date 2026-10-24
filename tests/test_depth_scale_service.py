import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from api.core.config import Settings, settings
from api.core.storage import LocalStorage
from api.schemas.ground_ref import DepthScaleCreate
from api.services import depth_scale_service as service, depth_service
from api.services.image_service import encode_image
from api.services.prediction_orchestrator import prepare_prediction_image
from api.services.vision_process_service import VisionError


def body(**changes):
    values = {"test_image": "registered/scene.png", "depth_model": "yolo26n-depth.pt",
              "known_distances": [{"x": 10, "y": 10, "distance_m": 3.0},
                                  {"x": 20, "y": 20, "distance_m": 5.0},
                                  {"x": 30, "y": 30, "distance_m": 7.0}]}
    return DepthScaleCreate.model_validate(dict(values, **changes))


@pytest.mark.parametrize("changes", [
    {"known_distances": []}, {"known_distances": [{}, {}, {}]},
    {"known_distances": [{"x": "1", "y": 2, "distance_m": 3}] * 3},
    {"known_distances": [{"x": 1.2, "y": 2, "distance_m": 3}] * 3},
    {"known_distances": [{"x": 1, "y": 2, "distance_m": -3}] * 3},
    {"known_distances": [{"x": 1, "y": 2, "distance_m": float("nan")}] * 3},
    {"known_distances": [{"x": 1, "y": 2, "distance_m": 3}] * 3},
    {"clip_min_m": 10, "clip_max_m": 5}, {"clip_min_m": float("nan")},
    {"clip_max_m": float("inf")}, {"depth_model": "../outside.pt"},
    {"depth_model": "https://host/model.pt"}, {"test_image": "/private/scene.png"},
    {"test_image": "cameras/../scene.png"}, {"test_image": "https://host/files/scene.png"},
    {"test_image": "cameras\\private.png"}, {"clip_min_m": 4},
])
def test_schema_rejects_bad_inputs(changes):
    with pytest.raises(ValidationError):
        body(**changes)


def raw_depth():
    return np.tile(np.arange(64, dtype=np.float32) / 10 + 0.01, (48, 1))


def test_linear_fit_and_signed_residuals():
    result = service.fit_depth_scale(raw_depth(), body())
    assert result["scale_s"] == pytest.approx(2)
    assert result["offset_b"] == pytest.approx(.98)
    assert result["max_residual_m"] < 1e-6
    np.testing.assert_allclose(result["residuals"], [0, 0, 0], atol=1e-6)


@pytest.mark.parametrize("kind", ["constant", "near_constant", "nan", "inf", "negative", "reverse", "poor", "constant_truth", "bounds"])
def test_rejected_depth_and_fits(kind):
    depth, request = raw_depth(), body()
    if kind == "constant":
        depth[:] = 1
    elif kind == "near_constant":
        depth[:] = 1
        depth[20, 20] += 1e-5
    elif kind in {"nan", "inf", "negative"}:
        depth[0, 0] = {"nan": np.nan, "inf": np.inf, "negative": -1}[kind]
    elif kind == "reverse":
        depth = depth[:, ::-1]
    elif kind == "poor":
        request.known_distances[1].distance_m = 15
    elif kind == "constant_truth":
        for point in request.known_distances:
            point.distance_m = 3
    else:
        request.known_distances[0].x = 64
    with pytest.raises(ValueError, match="^depth_scale_"):
        service.fit_depth_scale(depth, request)


def test_stored_scale_validation():
    result = service.fit_depth_scale(raw_depth(), body())
    scale = SimpleNamespace(**{key: value for key, value in result.items() if key != "depth_model"})
    service.validate_saved_scale(scale)
    for field, value in [("scale_s", -1), ("offset_b", np.nan), ("residuals", None), ("max_residual_m", 99)]:
        changed = SimpleNamespace(**dict(vars(scale), **{field: value}))
        with pytest.raises(ValueError):
            service.validate_saved_scale(changed)


def test_model_resolution_confines_symlinks_and_never_downloads(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(settings, "MODELS_DIR", str(models))
    good = models / "installed.pt"
    good.touch()
    assert depth_service.resolve_depth_model("installed.pt") == good
    outside = tmp_path / "outside.pt"
    outside.touch()
    (models / "escape.pt").symlink_to(outside)
    for name in ["missing.pt", "../outside.pt", str(outside), "escape.pt"]:
        with pytest.raises(ValueError, match="^depth_model_"):
            depth_service.resolve_depth_model(name)


def test_bounded_decode_closes_stream_and_checks_pixels(tmp_path):
    storage = LocalStorage(tmp_path)
    content = encode_image(np.zeros((300, 300, 3), np.uint8), ".png")
    storage.save("scene", io.BytesIO(content))
    with pytest.raises(ValueError, match="prediction_invalid_image"):
        service.read_image("scene", storage, Settings(_env_file=None, CAPTURE_MAX_PIXELS=65536))
    stream = io.BytesIO(b"invalid")
    fake = SimpleNamespace(get=lambda key: stream)
    with pytest.raises(ValueError, match="prediction_invalid_image"):
        service.read_image("scene", fake)
    assert stream.closed


def test_computation_prepares_same_image_as_prediction_and_bounds_process(monkeypatch, tmp_path):
    camera = SimpleNamespace(id=uuid4(), resolution_w=64, resolution_h=48)
    K = [[50, 0, 32], [0, 50, 24], [0, 0, 1]]
    calibration = SimpleNamespace(k_original=K, k_matrix=K, k_undist=K,
        resolution_w=64, resolution_h=48, dist_coeffs=[.3, -.1, 0, 0, 0])
    image = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    storage = LocalStorage(tmp_path / "storage")
    storage.save("registered/scene.png", io.BytesIO(encode_image(image, ".png")))
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26n-depth.pt").touch()
    monkeypatch.setattr(settings, "MODELS_DIR", str(models))
    monkeypatch.setattr(service, "get_storage", lambda: storage)
    received = []
    async def process(mode, payload, timeout):
        assert mode == "depth-scale" and timeout == settings.DEPTH_SCALE_TIMEOUT_SEC
        assert payload["model_path"] == str(models / "yolo26n-depth.pt")
        received.append(cv2.imread(payload["image_path"]))
        return service.fit_depth_scale(raw_depth(), body())
    runner = AsyncMock(side_effect=process)
    monkeypatch.setattr(service, "run_vision", runner)
    request = body()
    result = asyncio.run(service.compute_depth_scale(None, camera, calibration, None, request.test_image,
        request.depth_model, request.known_distances))
    assert result["scale_s"] == pytest.approx(2)
    np.testing.assert_array_equal(received[0], prepare_prediction_image(image, calibration, camera, "upload"))
    calibration.resolution_w = 128
    with pytest.raises(ValueError, match="prediction_resolution_mismatch"):
        asyncio.run(service.compute_depth_scale(None, camera, calibration, None, request.test_image,
            request.depth_model, request.known_distances))
    assert runner.await_count == 1


def test_vision_process_calls_actual_adapter(monkeypatch, tmp_path, capsys):
    import json
    import sys
    from api.workers import vision_process
    image = np.zeros((48, 64, 3), np.uint8)
    filename = tmp_path / "image.png"
    filename.write_bytes(encode_image(image, ".png"))
    def inference(received, model, imgsz, device):
        print("library diagnostic")
        assert received.shape == image.shape and model == "installed.pt" and device == "cpu"
        return raw_depth()
    monkeypatch.setattr(depth_service, "predict_depth", inference)
    payload = {"image_path": str(filename), "model_path": "installed.pt", "imgsz": 768,
               "body": body().model_dump(), "max_mean": .25, "max_residual": .5, "max_relative": .1}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(sys, "argv", ["vision_process", "depth-scale"])
    vision_process.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["scale_s"] == pytest.approx(2)
    assert "library diagnostic" in captured.err


def test_test_image_keys_are_ground_bound():
    camera, ground, image = uuid4(), uuid4(), uuid4()
    key = service.test_image_key(camera, ground, image)
    assert service.is_test_image_key(key, camera, ground)
    for invalid in [key.replace(str(ground), str(uuid4())), key + "/../secret", key.replace(str(image), "invalid")]:
        assert not service.is_test_image_key(invalid, camera, ground)


@pytest.mark.parametrize("kind", ["incomplete", "target_mismatch", "invalid_matrix"])
def test_invalid_intrinsics_cannot_define_different_depth_coordinates(kind):
    camera = SimpleNamespace(resolution_w=64, resolution_h=48)
    K = [[50, 0, 32], [0, 50, 24], [0, 0, 1]]
    calibration = SimpleNamespace(k_original=K, k_matrix=K, k_undist=K,
        resolution_w=64, resolution_h=48, dist_coeffs=[0]*5)
    if kind == "incomplete":
        calibration.resolution_w = None
    elif kind == "target_mismatch":
        calibration.k_undist = [[60, 0, 32], [0, 50, 24], [0, 0, 1]]
    else:
        calibration.k_matrix = [[0, 0, 0]] * 3
    with pytest.raises(ValueError, match="calibration_invalid_intrinsics"):
        service.prepare_image(np.zeros((48, 64, 3), np.uint8), calibration, camera)
