import io
from datetime import timezone, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.core.database import get_db
from api.core.security import verify_api_key
from api.core.storage import LocalStorage
from api.main import app
from api.routes import predictions
from api.services.rtsp_capture_service import CaptureError
from tests.test_calibration_capture_service import jpeg


@pytest.fixture
def setup(monkeypatch, tmp_path):
    camera = SimpleNamespace(id=uuid4(), rtsp_url="rtsp://camera.local/live", stream_enabled=True)
    key = SimpleNamespace(camera_id=None, scopes=["predict:read", "predict:write"])
    db = MagicMock(commit=AsyncMock(), rollback=AsyncMock(), refresh=AsyncMock(), scalar=AsyncMock(return_value=None))
    db.execute = AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: camera))
    job = SimpleNamespace(id=uuid4(), status="pending")
    submit = AsyncMock(return_value=job)
    queue = MagicMock()
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(predictions, "submit_prediction_job", submit)
    monkeypatch.setattr(predictions, "require_rtsp_camera", lambda _: "source-identity")
    monkeypatch.setattr(predictions, "latest_snapshot", AsyncMock())
    monkeypatch.setattr(predictions, "get_storage", lambda: storage)
    monkeypatch.setattr(predictions.celery_app, "send_task", queue)
    async def dependency():
        yield db
    app.dependency_overrides[get_db] = dependency
    app.dependency_overrides[verify_api_key] = lambda: key
    client = TestClient(app)
    yield SimpleNamespace(client=client, camera=camera, key=key, db=db, job=job, submit=submit, queue=queue,
                          storage=storage, url=f"/api/v1/cameras/{camera.id}/predict")
    client.close()
    app.dependency_overrides.clear()


def test_upload_unique_names_and_parameters(setup):
    paths = []
    for _ in range(2):
        response = setup.client.post(setup.url, data={"source_type": "upload", "volume_cell_m": "0.2", "volume_min_pts_per_cell": "3"},
            files={"file": ("../../private.jpg", jpeg(), "image/jpeg")})
        assert response.status_code == 202, response.text
        kwargs = setup.submit.call_args.kwargs
        paths.append(kwargs["source_path"])
        assert kwargs["volume_cell_m"] == 0.2 and kwargs["volume_min_pts_per_cell"] == 3
        assert "private" not in kwargs["source_path"] and ".." not in kwargs["source_path"]
        assert setup.storage.exists(kwargs["source_path"])
    assert paths[0] != paths[1]
    assert setup.queue.call_args.kwargs["queue"] == "prediction"
    assert setup.queue.call_args.kwargs["retry"] is False
    assert setup.db.commit.await_count == 2


@pytest.mark.parametrize(("field", "value"), [("seg_conf", "2"), ("volume_cell_m", "nan"),
    ("volume_min_pts_per_cell", "0"), ("max_frame_age_sec", "301"), ("seg_classes", '{}'),
    ("return_outputs", '[1]'), ("return_outputs", '["unknown"]'), ("seg_model", "../private.pt")])
def test_parameter_validation(setup, field, value):
    response = setup.client.post(setup.url, data={"source_type": "rtsp_frame", field: value})
    assert response.status_code == 422
    setup.queue.assert_not_called()


def test_invalid_image_and_prerequisites_do_not_leave_storage(setup):
    assert setup.client.post(setup.url, data={"source_type": "upload"}, files={"file": ("a.jpg", b"invalid")}).status_code == 400
    setup.submit.side_effect = ValueError("prediction_requires_calibration")
    response = setup.client.post(setup.url, data={"source_type": "upload"}, files={"file": ("a.jpg", jpeg())})
    assert response.status_code == 409
    assert list(setup.storage.root.rglob("*.image")) == []
    setup.queue.assert_not_called()


def test_rtsp_freshness_and_timestamp_contract(setup):
    response = setup.client.post(setup.url, data={"source_type": "rtsp_frame", "max_frame_age_sec": "5"})
    assert response.status_code == 202
    assert setup.submit.call_args.kwargs["source_hash"] == "source-identity"
    assert setup.submit.call_args.kwargs["max_frame_age_sec"] == 5
    assert setup.client.post(setup.url, data={"source_type": "rtsp_frame", "timestamp": datetime.now(timezone.utc).isoformat()}).status_code == 422
    assert setup.client.post(setup.url, data={"source_type": "latest_rtsp"}, files={"file": ("a.jpg", jpeg())}).status_code == 422
    predictions.latest_snapshot.side_effect = CaptureError("snapshot_stale")
    assert setup.client.post(setup.url, data={"source_type": "latest_rtsp"}).json()["detail"]["code"] == "snapshot_stale"


def test_queue_failure_is_safe_and_committed(setup):
    setup.queue.side_effect = RuntimeError("password@private-host")
    response = setup.client.post(setup.url, data={"source_type": "rtsp_frame"})
    assert response.status_code == 503
    assert "password" not in response.text
    assert setup.db.commit.await_count == 2
    assert setup.db.execute.call_args.args[0].is_update


@pytest.mark.parametrize("suffix", ["", "/outputs", "/outputs/mask", "/instances", "/volumes"])
def test_all_job_reads_are_camera_bound(setup, suffix):
    setup.key.camera_id = uuid4()
    assert setup.client.get(f"{setup.url}/{uuid4()}{suffix}").status_code == 403
    setup.key.camera_id = None
    if suffix:
        assert setup.client.get(f"{setup.url}/{uuid4()}{suffix}").status_code == 404


def test_openapi_prediction_multipart_and_snapshot_contract(setup):
    schema = setup.client.get("/openapi.json").json()
    path = schema["paths"]["/api/v1/cameras/{camera_id}/predict"]["post"]
    ref = path["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    fields = schema["components"]["schemas"][ref]["properties"]
    assert fields["max_frame_age_sec"]["minimum"] == 1
    assert fields["max_frame_age_sec"]["maximum"] == 300
    assert "202" in path["responses"]
    assert "201" in schema["paths"]["/api/v1/cameras/{camera_id}/snapshots"]["post"]["responses"]
    assert {"source_type", "file", "timestamp", "seg_classes", "return_outputs"}.issubset(fields)
    assert path["security"] == [{"ApiKeyAuth": []}]
    image = schema["paths"]["/api/v1/cameras/{camera_id}/snapshots/{snapshot_id}/image"]["get"]
    assert set(image["responses"]["200"]["content"]) == {"image/jpeg"}
