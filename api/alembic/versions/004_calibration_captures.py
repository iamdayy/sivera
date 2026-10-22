"""Persist bounded RTSP calibration capture sessions and images."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "004_calibration_captures"
down_revision = "003_camera_streams"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "calibration_capture_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_id", UUID(as_uuid=True), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("chessboard_cols", sa.Integer(), nullable=False),
        sa.Column("chessboard_rows", sa.Integer(), nullable=False),
        sa.Column("square_m", sa.Float(), nullable=False),
        sa.Column("interval_sec", sa.Float(), nullable=False),
        sa.Column("target_images", sa.Integer(), nullable=False),
        sa.Column("max_duration_sec", sa.Integer(), nullable=False),
        sa.Column("attempted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("captured", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("resolution_w", sa.Integer()),
        sa.Column("resolution_h", sa.Integer()),
        sa.Column("worker_token", UUID(as_uuid=True)),
        sa.Column("error_code", sa.String(64)),
        sa.Column("finish_reason", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        *[sa.Column(name, sa.DateTime(timezone=True)) for name in (
            "started_at", "completed_at", "heartbeat_at", "last_capture_at", "last_dispatched_at",
        )],
        sa.CheckConstraint("status IN ('queued','running','stopping','completed','stopped','expired','failed')", name="ck_capture_status"),
        sa.CheckConstraint("attempted >= 0 AND captured >= 0 AND failed_attempts >= 0 AND captured + failed_attempts <= attempted", name="ck_capture_counts"),
        sa.CheckConstraint("chessboard_cols BETWEEN 2 AND 30 AND chessboard_rows BETWEEN 2 AND 30 AND square_m > 0 AND square_m <= 1 AND interval_sec BETWEEN 1 AND 60 AND target_images BETWEEN 1 AND 100 AND max_duration_sec BETWEEN 1 AND 900", name="ck_capture_parameters"),
    )
    op.create_index("ix_calibration_capture_sessions_camera_id", "calibration_capture_sessions", ["camera_id"])
    op.create_index("ix_capture_status_deadline", "calibration_capture_sessions", ["status", "deadline_at"])
    op.create_index("uq_capture_active_camera", "calibration_capture_sessions", ["camera_id"], unique=True, postgresql_where=sa.text("status IN ('queued', 'running', 'stopping')"))
    op.create_table(
        "calibration_capture_images",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("calibration_capture_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("storage_key", sa.String(500), nullable=False, unique=True),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "sequence", name="uq_capture_image_sequence"),
        sa.CheckConstraint("sequence > 0 AND width > 0 AND height > 0 AND size_bytes > 0", name="ck_capture_image_dimensions"),
    )
    op.create_index("ix_calibration_capture_images_session_id", "calibration_capture_images", ["session_id"])


def downgrade() -> None:
    op.drop_table("calibration_capture_images")
    op.drop_table("calibration_capture_sessions")
