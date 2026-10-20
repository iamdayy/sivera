"""Unit tests for segmentation service mask utilities (pure functions)."""
import numpy as np
import pytest

from api.services.segment_service import clean_mask, rasterize, waste_mask_from


class TestCleanMask:
    def test_clean_mask_preserves_large_regions(self, sample_mask):
        cleaned = clean_mask(sample_mask, open_px=0, close_px=0, min_area_px=1)
        assert cleaned.shape == sample_mask.shape
        assert cleaned.dtype == bool
        assert cleaned.sum() > 0

    def test_clean_mask_removes_small_regions(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True  # single pixel
        cleaned = clean_mask(mask, open_px=0, close_px=0, min_area_px=10)
        assert cleaned.sum() == 0

    def test_clean_mask_empty(self):
        mask = np.zeros((10, 10), dtype=bool)
        cleaned = clean_mask(mask, open_px=0, close_px=0, min_area_px=1)
        assert cleaned.sum() == 0


class TestRasterize:
    def test_rasterize_triangle(self):
        polys = [np.array([[0.0, 0.0], [9.0, 0.0], [0.0, 9.0]], dtype=np.float32)]
        mask = rasterize(polys, (10, 10))
        assert mask.dtype == bool
        assert mask.sum() > 0
        # The origin should be filled
        assert mask[0, 0]

    def test_rasterize_empty(self):
        mask = rasterize([], (10, 10))
        assert mask.sum() == 0

    def test_rasterize_too_few_points(self):
        # A 2-point polygon is rasterized as a line (fills pixels along the line)
        polys = [np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)]
        mask = rasterize(polys, (10, 10))
        # Should not crash; line fills at least the endpoints
        assert mask.sum() >= 1


class TestWasteMaskFrom:
    def test_waste_mask_from_instances(self):
        instances = [
            {"cls": "Sampah_Berserakan", "conf": 0.9, "poly": np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32)},
            {"cls": "Other", "conf": 0.8, "poly": np.array([[30, 30], [40, 30], [40, 40], [30, 40]], dtype=np.float32)},
        ]
        waste_mask = waste_mask_from(instances, (50, 50), ["Sampah_Berserakan", "Sampah_Overload"])
        assert waste_mask.dtype == bool
        assert waste_mask.sum() > 0
        # Only the first instance should contribute
        assert waste_mask[15, 15]
        assert not waste_mask[35, 35]
