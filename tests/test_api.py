"""API tests: auth, scope, OpenAPI schema (no real database needed)."""
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from api.core.security import verify_api_key, require_scope, hash_key
from api.core.database import get_db
from api.main import app


class TestSecurity:
    def test_hash_key_is_deterministic(self):
        h1 = hash_key("test-key-123")
        h2 = hash_key("test-key-123")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_require_scope_denies_missing_scope(self):
        api_key = MagicMock()
        api_key.scopes = ["predict:read"]
        dependency = require_scope("camera:write")
        with pytest.raises(HTTPException) as exc:
            asyncio.run(dependency(api_key))
        assert exc.value.status_code == status.HTTP_403_FORBIDDEN

    def test_require_scope_allows_wildcard(self):
        api_key = MagicMock()
        api_key.scopes = ["*"]
        dependency = require_scope("camera:write")
        result = asyncio.run(dependency(api_key))
        assert result is api_key

    def test_require_scope_allows_exact_match(self):
        api_key = MagicMock()
        api_key.scopes = ["camera:write"]
        dependency = require_scope("camera:write")
        result = asyncio.run(dependency(api_key))
        assert result is api_key


class TestOpenAPI:
    def test_openapi_schema_generated(self):
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        # Security scheme should be present
        assert "securitySchemes" in schema["components"]
        assert "ApiKeyAuth" in schema["components"]["securitySchemes"]

    @pytest.mark.parametrize("connected", [True, False])
    def test_health_endpoint(self, connected):
        db = MagicMock(execute=AsyncMock(side_effect=None if connected else RuntimeError("private diagnostics")))
        async def dependency():
            yield db
        app.dependency_overrides[get_db] = dependency
        try:
            response = TestClient(app).get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == ("healthy" if connected else "degraded")
            assert response.json()["database"] == ("connected" if connected else "disconnected")
            assert "private" not in response.text
        finally:
            app.dependency_overrides.clear()

    def test_root_endpoint(self):
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_predict_route_registered(self):
        client = TestClient(app)
        response = client.post("/api/v1/cameras/00000000-0000-0000-0000-000000000000/predict",
                                data={"source_type": "upload"})
        assert response.status_code == 422

    def test_public_and_protected_security_matches_dependencies(self):
        schema = TestClient(app).get("/openapi.json").json()
        for path, operations in schema["paths"].items():
            for operation in operations.values():
                if not isinstance(operation, dict) or "operationId" not in operation:
                    continue
                assert operation["security"] == ([] if path in {"/", "/health"} else [{"ApiKeyAuth": []}])
                assert not any(parameter.get("in") == "header" and parameter["name"].lower() == "x-api-key"
                               for parameter in operation.get("parameters", []))
                if path not in {"/", "/health"}:
                    assert {"401", "403"} <= set(operation["responses"])

    def test_downloads_are_documented_as_binary_not_json_objects(self):
        schema = TestClient(app).get("/openapi.json").json()
        for path in ("/files/{path}", "/api/v1/cameras/{camera_id}/predict/{job_id}/outputs/{output_type}"):
            content = schema["paths"][path]["get"]["responses"]["200"]["content"]
            assert set(content) == {"image/jpeg", "image/png", "application/json", "application/octet-stream"}
            assert all(media["schema"] == {"type": "string", "format": "binary"} for media in content.values())
        base = "/api/v1/cameras/{camera_id}"
        for path in (base + "/snapshots/{snapshot_id}/image",
                     base + "/calibration-captures/{session_id}/images/{image_id}",
                     base + "/calibration-captures/{session_id}/images/{image_id}/preview"):
            content = schema["paths"][path]["get"]["responses"]["200"]["content"]
            assert content == {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}

    def test_health_and_root_response_schemas(self):
        schema = TestClient(app).get("/openapi.json").json()
        health = schema["components"]["schemas"]["HealthResponse"]["properties"]
        assert set(health["status"]["enum"]) == {"healthy", "degraded"}
        assert set(health["database"]["enum"]) == {"connected", "disconnected"}
        assert schema["paths"]["/"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/RootResponse")
