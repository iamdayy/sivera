"""Unit tests for calibration service (pure functions, no database)."""
import numpy as np
import pytest

from api.services.calibration_service import scale_K_for_resolution


class TestScaleKForResolution:
    def test_upscale(self):
        K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        orig_size = (640, 480)  # w, h
        target_size = (1280, 960)  # 2x upscale
        K_scaled = scale_K_for_resolution(K, orig_size, target_size)
        assert K_scaled[0, 0] == 1600.0  # fx * 2
        assert K_scaled[1, 1] == 1600.0  # fy * 2
        assert K_scaled[0, 2] == 640.0  # cx * 2
        assert K_scaled[1, 2] == 480.0  # cy * 2
        assert K_scaled[2, 2] == 1.0

    def test_downscale(self):
        K = np.array([[1600.0, 0.0, 640.0], [0.0, 1600.0, 480.0], [0.0, 0.0, 1.0]])
        orig_size = (1280, 960)
        target_size = (640, 480)  # 0.5x downscale
        K_scaled = scale_K_for_resolution(K, orig_size, target_size)
        assert K_scaled[0, 0] == 800.0
        assert K_scaled[1, 1] == 800.0
        assert K_scaled[0, 2] == 320.0
        assert K_scaled[1, 2] == 240.0

    def test_non_uniform_scale(self):
        K = np.array([[800.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
        orig_size = (640, 480)
        target_size = (1280, 480)  # 2x width, 1x height
        K_scaled = scale_K_for_resolution(K, orig_size, target_size)
        assert K_scaled[0, 0] == 1600.0  # fx * 2
        assert K_scaled[1, 1] == 400.0  # fy * 1
        assert K_scaled[0, 2] == 640.0  # cx * 2
        assert K_scaled[1, 2] == 240.0  # cy * 1
