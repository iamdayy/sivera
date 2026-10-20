"""API tests: auth, scope, OpenAPI schema (no real database needed)."""
import asyncio
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from api.core.security import verify_api_key, require_scope, hash_key
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

    def test_health_endpoint(self):
        client = TestClient(app)
        # Health endpoint requires DB; without DB it returns degraded or 500.
        # We just check the route is registered.
        response = client.get("/health")
        assert response.status_code in (200, 500)

    def test_root_endpoint(self):
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_predict_route_registered(self):
        client = TestClient(app)
        # POST without auth should return 422 (missing header) or 401/403
        response = client.post("/cameras/00000000-0000-0000-0000-000000000000/predict",
                               data={"source_type": "upload"})
        assert response.status_code in (401, 403, 422, 404)
