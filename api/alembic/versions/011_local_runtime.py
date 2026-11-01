"""Fence API-owned prediction runs and remove the redundant volume cache."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "011_local_runtime"
down_revision = "010_auto_predict_scheduler"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("auto_predict_deadline_at", sa.DateTime(timezone=True)))
    op.drop_column("cameras", "auto_predict_last_volume_liters")
    op.add_column("prediction_jobs", sa.Column("run_token", UUID(as_uuid=True)))
    op.add_column("prediction_jobs", sa.Column("deadline_at", sa.DateTime(timezone=True)))
    op.execute("""UPDATE cameras SET auto_predict_config =
        jsonb_set(auto_predict_config, '{source_type}', '"rtsp_frame"', true)::jsonb
        WHERE auto_predict_config IS NOT NULL AND jsonb_typeof(auto_predict_config) = 'object'""")


def downgrade() -> None:
    op.drop_column("prediction_jobs", "deadline_at")
    op.drop_column("prediction_jobs", "run_token")
    op.add_column("cameras", sa.Column("auto_predict_last_volume_liters", sa.Float()))
    op.drop_column("cameras", "auto_predict_deadline_at")
