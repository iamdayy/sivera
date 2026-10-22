"""Unit tests for volume estimation (pure functions, no database)."""
import numpy as np
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from api.services.volume_service import (
    estimate_volume_convexhull,
    estimate_per_instance,
    estimate_volume_grid,
)


class TestEstimateVolumeGrid:
    def test_known_grid_volume(self, synthetic_lifted_points):
        X, Y, H, uv = synthetic_lifted_points
        vol, footprint, grid_h, info = estimate_volume_grid(
            X, Y, H, cell_m=0.5, min_pts_per_cell=1
        )
        assert vol > 0.0
        assert footprint > 0.0
        # X span 0..1.5 at 0.5 cell => nx=3 cells; Y span 0..1 at 0.5 => ny=2 cells
        assert grid_h.shape == (2, 3)
        assert info["cell_m"] == 0.5
        assert info["nx"] == 3
        assert info["ny"] == 2

    def test_empty_points(self, empty_points):
        X, Y, H, uv = empty_points
        vol, footprint, grid_h, info = estimate_volume_grid(
            X, Y, H, cell_m=0.05, min_pts_per_cell=2
        )
        assert vol == 0.0
        assert footprint == 0.0
        assert grid_h.shape == (1, 1)
        assert info == {}

    def test_min_pts_per_cell_filter(self, synthetic_lifted_points):
        X, Y, H, uv = synthetic_lifted_points
        # With min_pts_per_cell=8 (all 8 points needed per cell), no cell qualifies
        vol, footprint, _, _ = estimate_volume_grid(
            X, Y, H, cell_m=0.5, min_pts_per_cell=8
        )
        assert vol == 0.0
        assert footprint == 0.0

    def test_single_point(self):
        X = np.array([0.0])
        Y = np.array([0.0])
        H = np.array([0.5])
        uv = np.array([[0, 0]], dtype=np.float64)
        vol, footprint, _, _ = estimate_volume_grid(
            X, Y, H, cell_m=0.05, min_pts_per_cell=1
        )
        assert vol > 0.0
        assert footprint == 0.05 ** 2

    def test_tiny_cells_reject_unbounded_grid(self):
        with pytest.raises(ValueError, match="prediction_volume_grid_too_large"):
            estimate_volume_grid(np.array([0., 1.]), np.array([0., 1.]), np.ones(2), cell_m=1e-9)


class TestEstimateVolumeConvexHull:
    def test_known_convexhull_volume(self, synthetic_lifted_points):
        X, Y, H, uv = synthetic_lifted_points
        vol, footprint = estimate_volume_convexhull(X, Y, H)
        assert vol > 0.0
        assert footprint > 0.0
        # footprint should be area of rectangle 1.5 x 1.0 = 1.5
        assert abs(footprint - 1.5) < 0.01

    def test_less_than_3_points(self):
        X = np.array([0.0, 0.5])
        Y = np.array([0.0, 0.0])
        H = np.array([0.1, 0.2])
        vol, footprint = estimate_volume_convexhull(X, Y, H)
        assert vol is None
        assert footprint is None

    def test_empty_points(self, empty_points):
        X, Y, H, uv = empty_points
        vol, footprint = estimate_volume_convexhull(X, Y, H)
        assert vol is None
        assert footprint is None

    def test_collinear_points_remain_missing(self):
        assert estimate_volume_convexhull(np.arange(4), np.arange(4), np.ones(4)) == (None, None)


class TestEstimatePerInstance:
    def test_per_instance_volumes(self, synthetic_lifted_points):
        X, Y, H, uv = synthetic_lifted_points
        # uv must be integer type for array indexing in estimate_per_instance
        uv_int = uv.astype(int)
        instances = [
            {"cls": "sampah_berserakan", "conf": 0.9, "polygon": [[100, 100], [140, 100], [140, 120], [100, 120]]},
            {"cls": "sampah_overload", "conf": 0.8, "polygon": [[100, 100], [120, 100], [120, 120], [100, 120]]},
        ]
        results = estimate_per_instance(
            instances,
            {"X": X, "Y": Y, "height": H, "uv": uv_int},
            img_shape=(200, 200),
            cell_m=0.05,
            min_pts_per_cell=1,
        )
        assert len(results) == 2
        for r in results:
            assert "cls" in r
            assert "conf" in r
            assert "volume_m3" in r
            assert "footprint_m2" in r
            assert "n_pts" in r
            assert r["volume_m3"] >= 0.0

    def test_empty_instances(self, synthetic_lifted_points):
        X, Y, H, uv = synthetic_lifted_points
        results = estimate_per_instance(
            [],
            {"X": X, "Y": Y, "height": H, "uv": uv},
            img_shape=(200, 200),
        )
        assert results == []


@pytest.mark.parametrize("points", [0, 2])
def test_unavailable_estimates_are_null_and_npz_nan(monkeypatch, tmp_path, points):
    from api.models.orm import PredictionJob, VolumeEstimate
    from api.core.storage import LocalStorage
    from api.services import volume_service
    from uuid import uuid4
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(volume_service, "get_storage", lambda: storage)
    db = MagicMock(flush=AsyncMock())
    job = PredictionJob(id=uuid4(), camera_id=uuid4())
    values = np.array([0., 1.])[:points]
    result = asyncio.run(volume_service.save_volume_results(db, job, values, values, np.ones(points),
        np.zeros((points, 2), int), [], (480, 640), .1, 3))
    assert result["volume_grid"] is None and result["volume_convex_hull"] is None
    assert not any(isinstance(call.args[0], VolumeEstimate) for call in db.add.call_args_list)
    with storage.get(result["volume_npz_key"]) as stream, np.load(stream, allow_pickle=False) as artifact:
        assert np.isnan(artifact["vol_grid"]) and np.isnan(artifact["vol_hull"])
