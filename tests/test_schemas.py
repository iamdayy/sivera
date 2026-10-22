"""Unit tests for Pydantic schema validation (no database)."""
from datetime import datetime
from uuid import uuid4, UUID

import pytest
from pydantic import ValidationError

from api.schemas.camera import CameraCreate, CameraUpdate
from api.schemas.prediction import PredictionJobCreate
from api.schemas.ground_ref import GroundReferenceBase
from api.schemas.calibration import CalibrationBase


class TestCameraSchema:
    def test_camera_create_valid(self):
        cam = CameraCreate(
            name="Cam-01",
            resolution_w=1920,
            resolution_h=1080,
            fps=25,
            latitude=-6.2088,
            longitude=106.8456,
        )
        assert cam.name == "Cam-01"
        assert cam.resolution_w == 1920
        assert cam.latitude == -6.2088

    def test_camera_create_invalid_resolution(self):
        with pytest.raises(ValidationError):
            CameraCreate(
                name="Cam-01",
                resolution_w=0,  # must be > 0
                resolution_h=1080,
            )

    def test_camera_create_invalid_latitude(self):
        with pytest.raises(ValidationError):
            CameraCreate(
                name="Cam-01",
                resolution_w=1920,
                resolution_h=1080,
                latitude=91.0,  # must be <= 90
            )

    def test_camera_update_partial(self):
        cam = CameraUpdate(name="Updated")
        assert cam.name == "Updated"
        assert cam.resolution_w is None


class TestPredictionJobSchema:
    def test_prediction_upload_valid(self):
        job = PredictionJobCreate(
            source_type="upload",
            seg_conf=0.5,
            seg_classes=["Sampah_Berserakan"],
        )
        assert job.source_type == "upload"
        assert job.seg_conf == 0.5

    def test_prediction_invalid_source_type(self):
        with pytest.raises(ValidationError):
            PredictionJobCreate(source_type="ftp")

    def test_prediction_seg_conf_out_of_range(self):
        with pytest.raises(ValidationError):
            PredictionJobCreate(source_type="upload", seg_conf=1.5)  # must be <= 1.0

    def test_prediction_volume_cell_m_out_of_range(self):
        with pytest.raises(ValidationError):
            PredictionJobCreate(source_type="upload", volume_cell_m=2.0)  # must be <= 1.0


class TestGroundReferenceSchema:
    def test_ground_ref_valid(self):
        ref = GroundReferenceBase(
            image_points=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            world_points=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            world_units="meter",
        )
        assert ref.world_units == "meter"

    def test_ground_ref_invalid_units(self):
        with pytest.raises(ValidationError):
            GroundReferenceBase(
                image_points=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
                world_points=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                world_units="feet",  # must be meter or metre
            )

    def test_ground_ref_insufficient_points(self):
        with pytest.raises(ValidationError):
            GroundReferenceBase(
                image_points=[[0.0, 0.0], [100.0, 0.0]],  # only 2, min 4
                world_points=[[0.0, 0.0], [2.0, 0.0]],
            )

    def test_ground_ref_mismatched_counts(self):
        with pytest.raises(ValidationError):
            GroundReferenceBase(
                image_points=[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
                world_points=[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],  # 3 vs 4
            )


class TestCalibrationSchema:
    def test_calibration_valid(self):
        calib = CalibrationBase(
            chessboard_cols=9,
            chessboard_rows=6,
            square_m=0.025,
            alpha=0.0,
        )
        assert calib.chessboard_cols == 9
        assert calib.square_m == 0.025

    def test_calibration_invalid_square_m(self):
        with pytest.raises(ValidationError):
            CalibrationBase(square_m=2.0)  # must be <= 1.0

    @pytest.mark.parametrize("values", [{"chessboard_rows": 2}, {"chessboard_cols": 31}, {"alpha": float("nan")}])
    def test_calibration_bounds_match_upload_route(self, values):
        with pytest.raises(ValidationError):
            CalibrationBase(**values)
