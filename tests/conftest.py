"""Pytest fixtures and configuration for SIVERA backend tests."""
import numpy as np
import pytest


@pytest.fixture
def synthetic_lifted_points():
    """Synthetic lifted ground points for volume/geometry tests.

    Returns X, Y, H, uv arrays for a small waste pile on the ground plane.
    Units: meters.
    """
    X = np.array([0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    Y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    H = np.array([0.1, 0.2, 0.15, 0.1, 0.1, 0.2, 0.15, 0.1], dtype=np.float64)
    uv = np.array(
        [[100, 100], [120, 100], [140, 100], [160, 100],
         [100, 120], [120, 120], [140, 120], [160, 120]],
        dtype=np.float64,
    )
    return X, Y, H, uv


@pytest.fixture
def empty_points():
    """Empty lifted points arrays."""
    X = np.array([], dtype=np.float64)
    Y = np.array([], dtype=np.float64)
    H = np.array([], dtype=np.float64)
    uv = np.zeros((0, 2), dtype=np.float64)
    return X, Y, H, uv


@pytest.fixture
def synthetic_calibration_points():
    """Synthetic image <-> world point correspondences for ground model tests."""
    image_points = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        dtype=np.float64,
    )
    world_points = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        dtype=np.float64,
    )
    K = np.array(
        [[800.0, 0.0, 50.0], [0.0, 800.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return image_points, world_points, K


@pytest.fixture
def sample_mask():
    """Small binary waste mask (True = waste)."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    return mask
