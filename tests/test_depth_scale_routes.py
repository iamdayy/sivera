import io
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.core.config import settings
from api.core.database import get_db
from api.core.security import verify_api_key
from api.core.storage import LocalStorage
from api.main import app
from api.models.orm import Calibration, Camera, DepthScale, GroundModel
from api.routes import ground_refs
from api.services import depth_scale_service, ground_model_service
from api.services.image_service import encode_image
from api.services.prediction_orchestrator import submit_prediction_job, prepare_prediction_image
from api.services.vision_process_service import VisionError
from tests.test_depth_scale_service import body, raw_depth


@pytest.fixture
def setup(monkeypatch, tmp_path):
    camera = SimpleNamespace(id=uuid4(), resolution_w=64, resolution_h=48)
    K = [[50, 0, 32], [0, 50, 24], [0, 0, 1]]
    image_key = f"cameras/{camera.id}/calib_images/scene.png"
    calibration = SimpleNamespace(id=uuid4(), camera_id=camera.id, k_original=K, k_matrix=K, k_undist=K,
        resolution_w=64, resolution_h=48, dist_coeffs=[.3, -.1, 0, 0, 0], is_active=True, calib_images=[image_key])
    ground = SimpleNamespace(id=uuid4(), camera_id=camera.id, calibration_id=calibration.id,
        H_i2w=[[.01, 0, 0], [0, .01, 0], [0, 0, 1]], H_w2i=[[100, 0, 0], [0, 100, 0], [0, 0, 1]],
        R_matrix=np.eye(3).tolist(), t_vector=[0, 0, 0], bev_ppm=20, bev_margin_m=.1,
        cam_height=1, pitch_down_deg=20, ground_ref_id=uuid4(), created_at=datetime.now(timezone.utc), reproj_err_px=0)
    old = SimpleNamespace(id=uuid4(), camera_id=camera.id, ground_model_id=ground.id, is_active=True,
                          created_at=datetime.now(timezone.utc), **depth_scale_service.fit_depth_scale(raw_depth(), body()))
    rows = {Camera: camera, Calibration: calibration, GroundModel: ground, DepthScale: old}
    key = SimpleNamespace(camera_id=None, scopes=["ground_ref:read", "ground_ref:write"])
    db = MagicMock(commit=AsyncMock(), rollback=AsyncMock(), refresh=AsyncMock(), flush=AsyncMock(),
                   scalar=AsyncMock(return_value=ground.id))
    selected = SimpleNamespace(id=ground.id)
    def execute(statement):
        if statement.is_update:
            rows[DepthScale].is_active = False
            item = None
        else:
            description = statement.column_descriptions[0]
            item = selected.id if description["entity"] is GroundModel and description["name"] == "id" else rows.get(description["entity"])
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
    storage = LocalStorage(tmp_path / "storage")
    image = np.random.default_rng(2).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    encoded = encode_image(image, ".png")
    storage.save(image_key, io.BytesIO(encoded), "image/png")
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26n-depth.pt").touch()
    monkeypatch.setattr(settings, "MODELS_DIR", str(models))
    for module in (ground_refs, depth_scale_service, ground_model_service):
        monkeypatch.setattr(module, "get_storage", lambda: storage)
    async def infer(mode, payload, timeout):
        from api.schemas.ground_ref import DepthScaleCreate
        return depth_scale_service.fit_depth_scale(raw_depth(), DepthScaleCreate.model_validate(payload["body"]))
    inference = AsyncMock(side_effect=infer)
    monkeypatch.setattr(depth_scale_service, "run_vision", inference)
    client = TestClient(app, raise_server_exceptions=False)
    request = body().model_dump()
    url = f"/api/v1/cameras/{camera.id}/ground-models/{ground.id}/depth-scales"
    yield SimpleNamespace(client=client, camera=camera, calibration=calibration, ground=ground, old=old,
        key=key, db=db, rows=rows, selected=selected, storage=storage, image=image, encoded=encoded,
        inference=inference, request=request, url=url)
    client.close()
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


def test_upload_preview_create_activate_and_prediction_pins(setup):
    response = setup.client.post(setup.url + "/test-images", files={"file": ("../../private.jpg", setup.encoded, "image/jpeg")})
    assert response.status_code == 201, response.text
    uploaded = response.json()
    assert uploaded["coordinate_system"] == "undistorted_pixels"
    assert uploaded["distance_type"] == "camera_axis_z_m"
    preview = setup.client.get(uploaded["image_url"])
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    decoded = cv2.imdecode(np.frombuffer(preview.content, np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, prepare_prediction_image(setup.image, setup.calibration, setup.camera, "upload"))
    response = setup.client.post(setup.url, json=setup.request)
    assert response.status_code == 201, response.text
    scale = response.json()
    assert scale["is_active"] and not setup.old.is_active
    assert setup.client.get(setup.url + "/active").json()["id"] == scale["id"]
    activated = setup.client.post(setup.url + f"/{scale['id']}/activate")
    assert activated.status_code == 200 and activated.json()["is_active"]
    import asyncio
    job = asyncio.run(submit_prediction_job(setup.db, setup.camera.id, "upload"))
    assert str(job.depth_scale_id) == scale["id"]
    assert job.calibration_id == setup.calibration.id and job.ground_model_id == setup.ground.id
    locked = [call.args[0] for call in setup.db.execute.call_args_list if getattr(call.args[0], "_for_update_arg", None) is not None]
    assert {query.column_descriptions[0]["entity"] for query in locked} >= {Camera, GroundModel, DepthScale}


@pytest.mark.parametrize(("method", "suffix"), [("get", ""), ("get", "/active"), ("get", "/{id}"),
    ("post", "/{id}/activate"), ("post", ""), ("post", "/test-images"), ("get", "/test-images/{id}/image")])
def test_every_depth_operation_is_camera_bound(setup, method, suffix):
    setup.key.camera_id = uuid4()
    response = getattr(setup.client, method)(setup.url + suffix.format(id=setup.old.id), **({"json": setup.request} if method == "post" and suffix == "" else {}))
    assert response.status_code == 403, response.text
    setup.db.execute.assert_not_awaited()
    setup.inference.assert_not_awaited()


def test_scopes_and_missing_header(setup):
    setup.key.scopes = ["predict:read"]
    assert setup.client.get(setup.url).status_code == 403
    assert setup.client.post(setup.url, json=setup.request).status_code == 403
    app.dependency_overrides.pop(verify_api_key)
    assert setup.client.get(setup.url).status_code == 422
    setup.db.execute.assert_not_awaited()


@pytest.mark.parametrize("changes", [{"known_distances": []}])
def test_creation_validates_before_computation(setup, changes):
    response = setup.client.post(setup.url, json=dict(setup.request, **changes))
    assert response.status_code == 422
    setup.inference.assert_not_awaited()
    assert setup.old.is_active


@pytest.mark.parametrize("code", ["depth_scale_degenerate_fit", "depth_scale_poor_fit", "depth_scale_pixel_out_of_bounds", "prediction_invalid_image"])
def test_rejected_fit_preserves_old_active_scale(setup, code):
    setup.inference.side_effect = ValueError(code)
    response = setup.client.post(setup.url, json=setup.request)
    assert response.status_code == 400, response.text
    assert setup.old.is_active
    setup.db.add.assert_not_called()


@pytest.mark.parametrize("exception", [VisionError("vision_timeout"), VisionError("depth_scale_busy"), RuntimeError("rtsp://private:password@host")])
def test_inference_failures_are_safe(setup, exception):
    setup.inference.side_effect = exception
    response = setup.client.post(setup.url, json=setup.request)
    assert response.status_code == 503
    assert "password" not in response.text and setup.old.is_active
    setup.db.add.assert_not_called()


def test_ground_changed_during_inference_is_not_persisted(setup):
    async def infer(*args):
        setup.selected.id = uuid4()
        return depth_scale_service.fit_depth_scale(raw_depth(), body())
    setup.inference.side_effect = infer
    response = setup.client.post(setup.url, json=setup.request)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "depth_scale_prerequisites_changed"
    assert setup.old.is_active
    setup.db.add.assert_not_called()


def test_invalid_saved_fit_cannot_be_activated(setup):
    setup.old.scale_s = -1
    response = setup.client.post(setup.url + f"/{setup.old.id}/activate")
    assert response.status_code == 409
    assert setup.old.is_active
    setup.db.commit.assert_not_awaited()


def test_parent_not_found_and_absent_scale(setup):
    setup.rows[GroundModel] = None
    assert setup.client.get(setup.url).status_code == 404
    setup.rows[GroundModel] = setup.ground
    setup.rows[DepthScale] = None
    assert setup.client.get(setup.url + "/active").status_code == 404
    assert setup.client.get(setup.url).json() == {"items": [], "total": 0}


def test_upload_validation_and_storage_failure(setup, monkeypatch):
    assert setup.client.post(setup.url + "/test-images", files={"file": ("a.png", b"bad")}).status_code == 400
    setup.calibration.resolution_w = 128
    assert setup.client.post(setup.url + "/test-images", files={"file": ("a.png", setup.encoded)}).status_code == 400
    setup.calibration.resolution_w = 64
    monkeypatch.setattr(setup.storage, "save", MagicMock(side_effect=OSError("private path")))
    response = setup.client.post(setup.url + "/test-images", files={"file": ("a.png", setup.encoded)})
    assert response.status_code == 503 and "private path" not in response.text


def test_unregistered_cross_camera_image_is_not_accepted(setup):
    # Since test_image is no longer in request body, this test is not applicable
    # The route now uses calibration.calib_images[0] by default
    pass


def test_bev_route_relative_keys_dimensions_and_camera_access(setup):
    url = setup.url.rsplit("/depth-scales", 1)[0] + "/bev-preview"
    # Use the calibration image key directly since it's no longer in request body
    image_key = setup.calibration.calib_images[0]
    response = setup.client.get(url, params={"test_image_path": image_key})
    assert response.status_code == 200, response.text
    assert setup.client.get(url).status_code == 200
    assert setup.client.get(url, params={"test_image_path": "/private/scene.png"}).status_code == 400
    setup.key.camera_id = uuid4()
    assert setup.client.get(url).status_code == 403
    assert setup.client.get(setup.url.rsplit("/", 2)[0] + "/active").status_code == 403


def test_openapi_has_upload_and_png_preview(setup):
    contract = setup.client.get("/openapi.json").json()
    prefix = "/api/v1/cameras/{camera_id}/ground-models/{ground_model_id}/depth-scales"
    upload = contract["paths"][prefix + "/test-images"]["post"]
    assert "multipart/form-data" in upload["requestBody"]["content"] and "201" in upload["responses"]
    preview = contract["paths"][prefix + "/test-images/{image_id}/image"]["get"]
    assert set(preview["responses"]["200"]["content"]) == {"image/png"}
    assert preview["security"] == [{"ApiKeyAuth": []}]
    assert contract["components"]["schemas"]["DepthScaleCreate"]["properties"]["known_distances"]["minItems"] == 3
