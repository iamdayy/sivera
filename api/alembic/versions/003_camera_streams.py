"""Persist desired camera stream state and owned WHEP sessions."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "003_camera_streams"
down_revision = "002_add_is_active_to_depth_scales"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("stream_enabled", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.create_table(
        "stream_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_id", UUID(as_uuid=True), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("owner_hash", sa.String(64), nullable=False),
        sa.Column("upstream_id", UUID(as_uuid=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_stream_sessions_camera_id", "stream_sessions", ["camera_id"])
    op.create_index("ix_stream_sessions_expires_at", "stream_sessions", ["expires_at"])


def downgrade() -> None:
    op.drop_table("stream_sessions")
    op.drop_column("cameras", "stream_enabled")
