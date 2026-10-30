"""Bind ground references to persisted, undistorted camera snapshots."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "008_ground_reference_sources"
down_revision = "007_preview_artifacts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ground_reference_sources",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_id", UUID(as_uuid=True), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("snapshot_id", UUID(as_uuid=True), sa.ForeignKey("camera_snapshots.id", deferrable=True, initially="DEFERRED"), nullable=False),
        sa.Column("calibration_id", UUID(as_uuid=True), sa.ForeignKey("calibrations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("storage_key", sa.String(500), nullable=False, unique=True),
        sa.Column("source_hash", sa.String(64), nullable=False),
        sa.Column("image_sha256", sa.String(64), nullable=False),
        sa.Column("preparation_version", sa.String(100), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("snapshot_id", "calibration_id", "preparation_version", name="uq_ground_source_preparation"),
        sa.CheckConstraint("width > 0 AND height > 0 AND size_bytes > 0", name="ck_ground_source_dimensions"),
    )
    op.create_index("ix_ground_reference_sources_camera_id", "ground_reference_sources", ["camera_id"])
    op.create_index("ix_ground_reference_sources_calibration_id", "ground_reference_sources", ["calibration_id"])
    op.add_column("ground_references", sa.Column("source_id", UUID(as_uuid=True), nullable=True))
    # Deferred checks allow camera/calibration cascades, but reject deleting a source
    # or snapshot that a surviving reference still needs.
    op.create_foreign_key("fk_ground_reference_source", "ground_references", "ground_reference_sources",
                          ["source_id"], ["id"], deferrable=True, initially="DEFERRED")
    op.create_index("ix_ground_references_source_id", "ground_references", ["source_id"])


def downgrade() -> None:
    op.drop_index("ix_ground_references_source_id", table_name="ground_references")
    op.drop_constraint("fk_ground_reference_source", "ground_references", type_="foreignkey")
    op.drop_column("ground_references", "source_id")
    op.drop_table("ground_reference_sources")
