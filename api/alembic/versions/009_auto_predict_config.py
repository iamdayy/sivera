"""Add auto_predict_config column to cameras."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "009_auto_predict_config"
down_revision = "008_ground_reference_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("auto_predict_config", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("cameras", "auto_predict_config")