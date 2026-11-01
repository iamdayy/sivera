"""Add scheduler metadata columns to cameras for auto-predict."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy import text


revision = "010_auto_predict_scheduler"
down_revision = "009_auto_predict_config"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add columns for scheduler metadata
    op.add_column("cameras", sa.Column("auto_predict_next_run", sa.DateTime(timezone=True), nullable=True))
    op.add_column("cameras", sa.Column("auto_predict_run_token", sa.String(64), nullable=True))
    op.add_column("cameras", sa.Column("auto_predict_last_run_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("cameras", sa.Column("auto_predict_last_volume_liters", sa.Float, nullable=True))
    op.add_column("cameras", sa.Column("auto_predict_last_saved_job_id", PGUUID(as_uuid=True), nullable=True))
    op.add_column("cameras", sa.Column("auto_predict_last_run_status", sa.String(30), nullable=True))
    op.add_column("cameras", sa.Column("auto_predict_last_run_error", sa.String(100), nullable=True))

    # Create index for finding cameras due for next run
    op.create_index(
        "ix_cameras_auto_predict_next_run",
        "cameras",
        ["auto_predict_next_run"],
        postgresql_where=sa.text("auto_predict_next_run IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_cameras_auto_predict_next_run", table_name="cameras")
    op.drop_column("cameras", "auto_predict_next_run")
    op.drop_column("cameras", "auto_predict_run_token")
    op.drop_column("cameras", "auto_predict_last_run_at")
    op.drop_column("cameras", "auto_predict_last_volume_liters")
    op.drop_column("cameras", "auto_predict_last_saved_job_id")
    op.drop_column("cameras", "auto_predict_last_run_status")
    op.drop_column("cameras", "auto_predict_last_run_error")