"""Bind depth fits to weights and inference preparation; preserve unknown legacy identity."""
from alembic import op
import sqlalchemy as sa


revision = "006_depth_model_identity"
down_revision = "005_capture_quality_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("depth_scales", sa.Column("model_identity", sa.JSON(), nullable=True))
    op.add_column("depth_scales", sa.Column("fit_provenance", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("depth_scales", "fit_provenance")
    op.drop_column("depth_scales", "model_identity")
