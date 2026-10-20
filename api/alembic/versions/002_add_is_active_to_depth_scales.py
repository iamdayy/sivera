"""Add is_active column to depth_scales.

Revision ID: 002_add_is_active_to_depth_scales
Revises: 001_initial
Create Date: 2026-10-20
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002_add_is_active_to_depth_scales'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'depth_scales',
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
    )
    op.create_index('ix_depth_scales_is_active', 'depth_scales', ['is_active'])


def downgrade() -> None:
    op.drop_index('ix_depth_scales_is_active', table_name='depth_scales')
    op.drop_column('depth_scales', 'is_active')
