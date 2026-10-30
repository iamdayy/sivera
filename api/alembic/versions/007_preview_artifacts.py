"""Register previews against their owning calibration or ground model."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "007_preview_artifacts"
down_revision = "006_depth_model_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "preview_artifacts",
        sa.Column("storage_key", sa.String(1024), primary_key=True),
        sa.Column("calibration_id", UUID(as_uuid=True), sa.ForeignKey("calibrations.id", ondelete="CASCADE")),
        sa.Column("ground_model_id", UUID(as_uuid=True), sa.ForeignKey("ground_models.id", ondelete="CASCADE")),
        sa.CheckConstraint("(calibration_id IS NULL) <> (ground_model_id IS NULL)", name="ck_preview_one_parent"),
    )
    op.create_index("ix_preview_artifacts_calibration_id", "preview_artifacts", ["calibration_id"])
    op.create_index("ix_preview_artifacts_ground_model_id", "preview_artifacts", ["ground_model_id"])


def downgrade() -> None:
    op.drop_table("preview_artifacts")
