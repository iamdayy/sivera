import sys
sys.path.insert(0, '/app')
import asyncio
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import cv2
import io
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from api.core.config import settings
from api.core.storage import LocalStorage
from api.main import app
from api.models.orm import Calibration, Camera, DepthScale, GroundModel, GroundReference
from api.routes import ground_refs
from api.services import depth_scale_service, ground_model_service
from api.services.image_service import encode_image
from fastapi.testclient import TestClient
from api.core.database import get_db
from api.core.security import verify_api_key

# Setup
camera = SimpleNamespace(id=uuid4(), resolution_w=64, resolution_h=48)
K = [[50, 0, 32], [0, 50, 24], [0, 0, 1]]
image_key = f'cameras/{camera.id}/calib_images/scene.png'
calibration = SimpleNamespace(id=uuid4(), camera_id=camera.id, k_original=K, k_matrix=K, k_undist=K,
    resolution_w=64, resolution_h=48, dist_coeffs=[.3, -.1, 0, 0, 0], is_active=True, calib_images=[image_key])
ground_ref_id = uuid4()
ground = SimpleNamespace(id=uuid4(), camera_id=camera.id, calibration_id=calibration.id,
    H_i2w=[[.01, 0, 0], [0, .01, 0], [0, 0, 1]], H_w2i=[[100, 0, 0], [0, 100, 0], [0, 0, 1]],
    R_matrix=np.eye(3).tolist(), t_vector=[0, 0, 0], bev_ppm=20, bev_margin_m=.1,
    cam_height=1, pitch_down_deg=20, ground_ref_id=ground_ref_id, created_at=datetime.now(timezone.utc), reproj_err_px=0)
ground_ref = SimpleNamespace(id=ground_ref_id, camera_id=camera.id, calibration_id=calibration.id,
    version=1, source_id=None, image_points=[[10,10],[50,10],[50,30],[10,30]],
    world_points=[[0,0],[1,0],[1,1],[0,1]], world_units='meter', description=None, is_active=True)

# Create depth scale data
image_points = np.array([[10,10],[50,10],[50,30]], dtype=np.float64)
world_points = np.array([[0,0],[1,0],[1,1]], dtype=np.float64)
known_distances = [{'p1': 0, 'p2': 1, 'distance_m': 1.0}, {'p1': 1, 'p2': 2, 'distance_m': 1.0}, {'p1': 0, 'p2': 2, 'distance_m': 2.0}]
depth_scale_data = depth_scale_service.fit_depth_scale(
    np.vstack([image_points.T, world_points.T]),  # This is wrong format
    {'image_points': image_points.tolist(), 'world_points': world_points.tolist(), 'known_distances': known_distances}
)

old = SimpleNamespace(id=uuid4(), camera_id=camera.id, ground_model_id=ground.id, is_active=True,
                      created_at=datetime.now(timezone.utc), **depth_scale_data)

rows = {Camera: camera, Calibration: calibration, GroundModel: ground, DepthScale: old, GroundReference: ground_ref}
key = SimpleNamespace(camera_id=None, scopes=['ground_ref:read', 'ground_ref:write'])
db = MagicMock(commit=AsyncMock(), rollback=AsyncMock(), refresh=AsyncMock(), flush=AsyncMock(),
               scalar=AsyncMock(return_value=ground.id))
selected = SimpleNamespace(id=ground.id)
def execute(statement):
    if statement.is_update:
        rows[DepthScale].is_active = False
        item = None
    else:
        description = statement.column_descriptions[0]
        item = selected.id if description['entity'] is GroundModel and description['name'] == 'id' else rows.get(description['entity'])
    return SimpleNamespace(scalar_one_or_none=lambda: item,
                           scalars=lambda: SimpleNamespace(all=lambda: [] if item is None else [item]))
db.execute = AsyncMock(side_effect=execute)
def add(record):
    record.id = uuid4()
    if isinstance(record, DepthScale):
        record.created_at = datetime.now(timezone.utc)
        rows[DepthScale] = record
db.add.side_effect = add

async def dependency():
    yield db

previous = app.dependency_overrides.copy()
app.dependency_overrides[get_db] = dependency
app.dependency_overrides[verify_api_key] = lambda: key

with tempfile.TemporaryDirectory() as tmp_dir:
    storage = LocalStorage(Path(tmp_dir) / 'storage')
    image = np.random.default_rng(2).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    encoded = encode_image(image, '.png')
    storage.save(image_key, io.BytesIO(encoded), 'image/png')
    models = Path(tmp_dir) / 'models'
    models.mkdir()
    (models / 'yolo26n-depth.pt').touch()
    settings.MODELS_DIR = str(models)
    for module in (ground_refs, depth_scale_service, ground_model_service):
        module.get_storage = lambda: storage
    async def infer(mode, payload, timeout):
        return {'scale_s': 1.0, 'scale_d': 0.0, 'rmse_m': 0.01, 'points_used': 3, 'fit_quality': 'good', 'raw_depth': payload.get('depth_values', []), 'fitted_depth': payload.get('depth_values', []), 'clipped': False}
    inference = AsyncMock(side_effect=infer)
    depth_scale_service.run_vision = inference
    client = TestClient(app, raise_server_exceptions=False)
    url = f'/api/v1/cameras/{camera.id}/ground-models/{ground.id}/depth-scales'
    bev_url = url.rsplit('/depth-scales', 1)[0] + '/bev-preview'
    response = client.get(bev_url, params={'test_image_path': image_key})
    print('STATUS:', response.status_code)
    print('RESPONSE:', response.text)
    if response.status_code == 200:
        print('SUCCESS!')
    else:
        print('FAILED!')
    client.close()
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)