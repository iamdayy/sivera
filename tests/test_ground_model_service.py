"""Unit tests for ground model geometry (pure functions, no database)."""
import asyncio
from uuid import uuid4

import cv2
import numpy as np
import pytest

from api.core.storage import LocalStorage
from api.models.orm import Calibration, Camera, GroundModel
from api.services import ground_model_service as service
from api.services.ground_model_service import (
    compute_bev,
    compute_ground_model,
    to_world,
)


class TestComputeGroundModel:
    def test_valid_points(self, synthetic_calibration_points):
        image_points, world_points, K = synthetic_calibration_points
        result = compute_ground_model(image_points, world_points, K)
        assert "H_w2i" in result
        assert "H_i2w" in result
        assert "R" in result
        assert "t" in result
        assert "K" in result
        assert "cam_height" in result
        assert "pitch_down_deg" in result
        assert result["cam_height"] > 0
        assert result["reproj_err_mean"] >= 0
        # H_i2w @ H_w2i should be identity
        H_i2w = result["H_i2w"]
        H_w2i = result["H_w2i"]
        product = H_i2w @ H_w2i
        np.testing.assert_allclose(product, np.eye(3), atol=1e-6)

    def test_insufficient_points(self):
        image_points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        world_points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        K = np.eye(3)
        with pytest.raises(ValueError):
            compute_ground_model(image_points, world_points, K)

    def test_mismatched_point_counts(self):
        image_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        world_points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        K = np.eye(3)
        with pytest.raises(ValueError):
            compute_ground_model(image_points, world_points, K)


class TestToWorld:
    def test_to_world_roundtrip(self, synthetic_calibration_points):
        image_points, world_points, K = synthetic_calibration_points
        result = compute_ground_model(image_points, world_points, K)
        H_i2w = result["H_i2w"]
        # Convert image points to world
        world_pts, valid = to_world(H_i2w, image_points)
        assert valid.all()
        np.testing.assert_allclose(world_pts, world_points, atol=1e-4)


class TestComputeBEV:
    def test_bev_output_shape(self, synthetic_calibration_points, sample_mask):
        image_points, world_points, K = synthetic_calibration_points
        gm = compute_ground_model(image_points, world_points, K)
        gm_dict = {
            "H_i2w": gm["H_i2w"],
            "H_w2i": gm["H_w2i"],
            "R": gm["R"],
            "t": gm["t"],
            "K": gm["K"],
            "cam_height": gm["cam_height"],
            "pitch_down_deg": gm["pitch_down_deg"],
        }
        # Create a small test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:70, 30:70] = [255, 0, 0]
        bev_img, bev_mask, bev_overlay, bev_info = compute_bev(
            image, sample_mask.astype(np.uint8), gm_dict, ppm=100, margin_m=0.5
        )
        assert bev_img.shape[2] == 3
        assert bev_mask.shape == bev_img.shape[:2]
        assert bev_overlay.shape == bev_img.shape
        assert "width_px" in bev_info
        assert "height_px" in bev_info
        assert "ppm" in bev_info
        assert bev_info["ppm"] == 100

    def test_empty_mask(self, synthetic_calibration_points):
        image_points, world_points, K = synthetic_calibration_points
        gm = compute_ground_model(image_points, world_points, K)
        gm_dict = {
            "H_i2w": gm["H_i2w"],
            "H_w2i": gm["H_w2i"],
            "R": gm["R"],
            "t": gm["t"],
            "K": gm["K"],
            "cam_height": gm["cam_height"],
            "pitch_down_deg": gm["pitch_down_deg"],
        }
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        bev_img, bev_mask, bev_overlay, bev_info = compute_bev(
            image, empty_mask, gm_dict, ppm=100, margin_m=0.5
        )
        assert bev_img.shape[2] == 3


def test_bev_preview_encodes_images_with_persisted_matrices(monkeypatch, tmp_path):
    storage = LocalStorage(tmp_path / "artifacts")
    monkeypatch.setattr(service, "get_storage", lambda: storage)
    camera = Camera(id=uuid4(), resolution_w=100, resolution_h=100)
    calibration = Calibration(k_matrix=np.eye(3).tolist(), k_original=None, resolution_w=None, resolution_h=None)
    ground = GroundModel(H_i2w=[[.01, 0, 0], [0, .01, 0], [0, 0, 1]],
                         H_w2i=[[100, 0, 0], [0, 100, 0], [0, 0, 1]],
                         R_matrix=np.eye(3).tolist(), t_vector=[0, 0, 0],
                         bev_ppm=20, bev_margin_m=.1)
    # Save test image to storage and use storage key
    test_image = np.full((100, 100, 3), 80, np.uint8)
    test_key = "cameras/test/bev_test.jpg"
    import io
    success, buf = cv2.imencode(".jpg", test_image)
    storage.save(test_key, io.BytesIO(buf.tobytes()), "image/jpeg")
    preview = asyncio.run(service.generate_bev_preview(None, camera, calibration, ground, test_key))
    assert preview["bev_info"]["ppm"] == 20
    for field in ("bev_image_url", "bev_mask_url", "bev_overlay_url"):
        key = preview[field].split("/files/", 1)[-1]
        with storage.get(key) as stream:
            content = stream.read()
        assert content.startswith(b"\xff\xd8")
        decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (preview["bev_info"]["height_px"], preview["bev_info"]["width_px"])
