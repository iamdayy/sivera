"""Capture quality, explicit calibration intrinsics and RTSP snapshots."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "005_capture_quality_snapshots"
down_revision = "004_calibration_captures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name in ("k_original", "k_undist", "per_view_errors"):
        op.add_column("calibrations", sa.Column(name, sa.JSON()))
    for name in ("resolution_w", "resolution_h"):
        op.add_column("calibrations", sa.Column(name, sa.Integer()))
    op.add_column("calibrations", sa.Column("alpha", sa.Float()))
    for name in ("accepted", "rejected", "selection_version"):
        op.add_column("calibration_capture_sessions", sa.Column(name, sa.Integer(), nullable=False, server_default="0"))
    op.add_column("calibration_capture_sessions", sa.Column("auto_calibrate", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("calibration_capture_sessions", sa.Column("source_hash", sa.String(64)))
    op.add_column("calibration_capture_sessions", sa.Column("rejection_counts", sa.JSON(), nullable=False, server_default="{}"))
    op.add_column("calibration_capture_sessions", sa.Column("compute_status", sa.String(20), nullable=False, server_default="none"))
    op.add_column("calibration_capture_sessions", sa.Column("compute_token", UUID(as_uuid=True)))
    for name in ("compute_requested_at", "compute_started_at", "compute_dispatched_at", "compute_completed_at"):
        op.add_column("calibration_capture_sessions", sa.Column(name, sa.DateTime(timezone=True)))
    op.add_column("calibration_capture_sessions", sa.Column("compute_error_code", sa.String(64)))
    op.add_column("calibration_capture_sessions", sa.Column("result_calibration_id", UUID(as_uuid=True)))
    op.create_foreign_key("fk_capture_result_calibration", "calibration_capture_sessions", "calibrations", ["result_calibration_id"], ["id"], ondelete="SET NULL")
    op.create_check_constraint("ck_capture_selection_counts", "calibration_capture_sessions", "accepted >= 0 AND rejected >= 0 AND accepted + rejected <= captured")
    op.create_check_constraint("ck_capture_compute_status", "calibration_capture_sessions", "compute_status IN ('none','queued','running','done','failed')")
    op.add_column("calibration_capture_images", sa.Column("quality_status", sa.String(20), nullable=False, server_default="unassessed"))
    op.add_column("calibration_capture_images", sa.Column("rejection_reason", sa.String(64)))
    op.add_column("calibration_capture_images", sa.Column("sharpness", sa.Float()))
    for name in ("corners", "pose_descriptor"):
        op.add_column("calibration_capture_images", sa.Column(name, sa.JSON()))
    op.add_column("prediction_jobs", sa.Column("max_frame_age_sec", sa.Integer(), nullable=False, server_default="30"))
    op.add_column("prediction_jobs", sa.Column("source_hash", sa.String(64)))
    op.add_column("prediction_jobs", sa.Column("volume_cell_m", sa.Float(), nullable=False, server_default="0.05"))
    op.add_column("prediction_jobs", sa.Column("volume_min_pts_per_cell", sa.Integer(), nullable=False, server_default="2"))
    op.add_column("prediction_jobs", sa.Column("return_outputs", sa.JSON()))
    op.create_table("camera_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_id", UUID(as_uuid=True), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_hash", sa.String(64), nullable=False),
        sa.Column("storage_key", sa.String(500), nullable=False, unique=True),
        sa.Column("width", sa.Integer(), nullable=False), sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False), sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("width > 0 AND height > 0 AND size_bytes > 0", name="ck_snapshot_dimensions"),
    )
    op.create_index("ix_camera_snapshots_camera_id", "camera_snapshots", ["camera_id"])
    op.create_index("ix_camera_snapshots_captured_at", "camera_snapshots", ["captured_at"])


def downgrade() -> None:
    op.drop_table("camera_snapshots")
    for name in ("return_outputs", "source_hash", "max_frame_age_sec", "volume_cell_m", "volume_min_pts_per_cell"):
        op.drop_column("prediction_jobs", name)
    for name in ("quality_status", "rejection_reason", "sharpness", "corners", "pose_descriptor"):
        op.drop_column("calibration_capture_images", name)
    op.drop_constraint("fk_capture_result_calibration", "calibration_capture_sessions", type_="foreignkey")
    op.drop_constraint("ck_capture_selection_counts", "calibration_capture_sessions", type_="check")
    op.drop_constraint("ck_capture_compute_status", "calibration_capture_sessions", type_="check")
    for name in ("accepted", "rejected", "selection_version", "auto_calibrate", "source_hash", "rejection_counts", "compute_status", "compute_token", "compute_requested_at", "compute_started_at", "compute_dispatched_at", "compute_completed_at", "compute_error_code", "result_calibration_id"):
        op.drop_column("calibration_capture_sessions", name)
    for name in ("k_original", "k_undist", "per_view_errors", "resolution_w", "resolution_h", "alpha"):
        op.drop_column("calibrations", name)
