"""Initial migration: create all tables.

Revision ID: 001_initial
Revises: 
Create Date: 2026-10-19
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
import geoalchemy2


# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable PostGIS extension
    op.execute('CREATE EXTENSION IF NOT EXISTS postgis;')
    
    # cameras table
    op.create_table(
        'cameras',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('rtsp_url', sa.String(500), nullable=True),
        sa.Column('location', geoalchemy2.Geography(geometry_type='POINT', srid=4326), nullable=True),
        sa.Column('resolution_w', sa.Integer(), nullable=False),
        sa.Column('resolution_h', sa.Integer(), nullable=False),
        sa.Column('fps', sa.Integer(), default=25, nullable=False),
        sa.Column('roi_polygon', geoalchemy2.Geography(geometry_type='POLYGON', srid=4326), nullable=True),
        sa.Column('metadata', sa.JSON(), default={}, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('ix_cameras_name', 'cameras', ['name'])
    
    # calibrations table
    op.create_table(
        'calibrations',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('camera_id', UUID(as_uuid=True), sa.ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('k_matrix', sa.JSON(), nullable=False),
        sa.Column('dist_coeffs', sa.JSON(), nullable=False),
        sa.Column('rms_reproj', sa.Float(), nullable=True),
        sa.Column('coverage_w', sa.Float(), nullable=True),
        sa.Column('coverage_h', sa.Float(), nullable=True),
        sa.Column('chessboard_cols', sa.Integer(), nullable=True),
        sa.Column('chessboard_rows', sa.Integer(), nullable=True),
        sa.Column('square_m', sa.Float(), nullable=True),
        sa.Column('calib_images', sa.JSON(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint('uq_calibration_camera_version', 'calibrations', ['camera_id', 'version'])
    op.create_index('ix_calibrations_camera_active', 'calibrations', ['camera_id', 'is_active'])
    op.create_index('ix_calibrations_camera_id', 'calibrations', ['camera_id'])
    
    # ground_references table
    op.create_table(
        'ground_references',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('camera_id', UUID(as_uuid=True), sa.ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False),
        sa.Column('calibration_id', UUID(as_uuid=True), sa.ForeignKey('calibrations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('image_points', sa.JSON(), nullable=False),
        sa.Column('world_points', sa.JSON(), nullable=False),
        sa.Column('world_units', sa.String(10), default='meter', nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_unique_constraint('uq_ground_ref_camera_version', 'ground_references', ['camera_id', 'version'])
    op.create_index('ix_ground_refs_camera_active', 'ground_references', ['camera_id', 'is_active'])
    op.create_index('ix_ground_refs_camera_id', 'ground_references', ['camera_id'])
    op.create_index('ix_ground_refs_calibration_id', 'ground_references', ['calibration_id'])
    
    # ground_models table
    op.create_table(
        'ground_models',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('camera_id', UUID(as_uuid=True), sa.ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False),
        sa.Column('calibration_id', UUID(as_uuid=True), sa.ForeignKey('calibrations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('ground_ref_id', UUID(as_uuid=True), sa.ForeignKey('ground_references.id', ondelete='CASCADE'), nullable=False),
        sa.Column('R_matrix', sa.JSON(), nullable=False),
        sa.Column('t_vector', sa.JSON(), nullable=False),
        sa.Column('H_i2w', sa.JSON(), nullable=False),
        sa.Column('H_w2i', sa.JSON(), nullable=False),
        sa.Column('cam_height', sa.Float(), nullable=True),
        sa.Column('pitch_down_deg', sa.Float(), nullable=True),
        sa.Column('reproj_err_px', sa.Float(), nullable=True),
        sa.Column('bev_ppm', sa.Integer(), default=200, nullable=False),
        sa.Column('bev_margin_m', sa.Float(), default=1.0, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_ground_models_camera_id', 'ground_models', ['camera_id'])
    op.create_index('ix_ground_models_calibration_id', 'ground_models', ['calibration_id'])
    op.create_index('ix_ground_models_ground_ref_id', 'ground_models', ['ground_ref_id'])
    
    # depth_scales table
    op.create_table(
        'depth_scales',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('camera_id', UUID(as_uuid=True), sa.ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False),
        sa.Column('ground_model_id', UUID(as_uuid=True), sa.ForeignKey('ground_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('depth_model', sa.String(100), nullable=False),
        sa.Column('scale_s', sa.Float(), nullable=False),
        sa.Column('offset_b', sa.Float(), nullable=False),
        sa.Column('residuals', sa.JSON(), nullable=True),
        sa.Column('mean_residual_m', sa.Float(), nullable=True),
        sa.Column('max_residual_m', sa.Float(), nullable=True),
        sa.Column('clip_min_m', sa.Float(), default=0.1, nullable=False),
        sa.Column('clip_max_m', sa.Float(), default=50.0, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_depth_scales_camera_id', 'depth_scales', ['camera_id'])
    op.create_index('ix_depth_scales_ground_model_id', 'depth_scales', ['ground_model_id'])
    
    # prediction_jobs table
    op.create_table(
        'prediction_jobs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('camera_id', UUID(as_uuid=True), sa.ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False),
        sa.Column('calibration_id', UUID(as_uuid=True), sa.ForeignKey('calibrations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('ground_model_id', UUID(as_uuid=True), sa.ForeignKey('ground_models.id', ondelete='CASCADE'), nullable=False),
        sa.Column('depth_scale_id', UUID(as_uuid=True), sa.ForeignKey('depth_scales.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(20), default='pending', nullable=False),
        sa.Column('source_type', sa.String(20), nullable=False),
        sa.Column('source_path', sa.String(500), nullable=True),
        sa.Column('source_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('seg_model', sa.String(100), nullable=True),
        sa.Column('seg_conf', sa.Float(), default=0.25, nullable=False),
        sa.Column('seg_classes', sa.JSON(), nullable=True),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error_msg', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_jobs_camera_status', 'prediction_jobs', ['camera_id', 'status'])
    op.create_index('ix_jobs_created', 'prediction_jobs', ['created_at'])
    op.create_index('ix_jobs_camera_id', 'prediction_jobs', ['camera_id'])
    op.create_index('ix_jobs_status', 'prediction_jobs', ['status'])
    
    # prediction_outputs table
    op.create_table(
        'prediction_outputs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', UUID(as_uuid=True), sa.ForeignKey('prediction_jobs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('output_type', sa.String(50), nullable=False),
        sa.Column('file_path', sa.String(500), nullable=False),
        sa.Column('mime_type', sa.String(50), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_outputs_job_id', 'prediction_outputs', ['job_id'])
    
    # detected_instances table
    op.create_table(
        'detected_instances',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', UUID(as_uuid=True), sa.ForeignKey('prediction_jobs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('cls_name', sa.String(50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('polygon_px', sa.JSON(), nullable=False),
        sa.Column('polygon_world', sa.JSON(), nullable=True),
        sa.Column('volume_m3', sa.Float(), nullable=True),
        sa.Column('footprint_m2', sa.Float(), nullable=True),
        sa.Column('height_mean_m', sa.Float(), nullable=True),
        sa.Column('height_max_m', sa.Float(), nullable=True),
        sa.Column('n_pixels', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_instances_job_id', 'detected_instances', ['job_id'])
    
    # volume_estimates table
    op.create_table(
        'volume_estimates',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', UUID(as_uuid=True), sa.ForeignKey('prediction_jobs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('method', sa.String(30), nullable=False),
        sa.Column('volume_m3', sa.Float(), nullable=False),
        sa.Column('volume_liter', sa.Float(), nullable=False),
        sa.Column('footprint_m2', sa.Float(), nullable=False),
        sa.Column('cell_m', sa.Float(), nullable=True),
        sa.Column('height_mean_m', sa.Float(), nullable=True),
        sa.Column('height_median_m', sa.Float(), nullable=True),
        sa.Column('n_cells', sa.Integer(), nullable=True),
        sa.Column('n_points', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_volume_estimates_job_id', 'volume_estimates', ['job_id'])
    
    # api_keys table
    op.create_table(
        'api_keys',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('camera_id', UUID(as_uuid=True), sa.ForeignKey('cameras.id', ondelete='SET NULL'), nullable=True),
        sa.Column('key_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('scopes', sa.JSON(), default=['predict', 'read'], nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_api_keys_camera_id', 'api_keys', ['camera_id'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'])


def downgrade() -> None:
    op.drop_table('api_keys')
    op.drop_table('volume_estimates')
    op.drop_table('detected_instances')
    op.drop_table('prediction_outputs')
    op.drop_table('prediction_jobs')
    op.drop_table('depth_scales')
    op.drop_table('ground_models')
    op.drop_table('ground_references')
    op.drop_table('calibrations')
    op.drop_table('cameras')
    op.execute('DROP EXTENSION IF EXISTS postgis;')