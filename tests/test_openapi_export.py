import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "api/scripts/export_openapi.py"


def export_contract(directory: Path, env: dict[str, str]) -> bytes:
    destination = directory / "openapi.json"
    subprocess.run(
        [sys.executable, str(EXPORTER), str(destination)],
        cwd=directory, env=env, capture_output=True, text=True, check=True,
    )
    return destination.read_bytes()


def test_export_ignores_environment_and_dotenv(tmp_path):
    env = dict(os.environ, TMPDIR=str(tmp_path))
    baseline = export_contract(tmp_path, env)
    overrides = {
        "APP_NAME": "private-title-sentinel",
        "APP_VERSION": "private-version-sentinel",
        "DEFAULT_CHESSBOARD_COLS": "invalid-integer",
        "DEFAULT_SEG_MODEL": "private-model-sentinel",
        "DATABASE_URL": "invalid-database-url",
        "SECRET_KEY": "private-secret-sentinel",
        "VALID_API_KEYS": "private-api-key-sentinel",
        "MEDIAMTX_API_PASSWORD": "private-password-sentinel",
        "MEDIAMTX_READ_PASSWORD": "private-reader-sentinel",
    }
    (tmp_path / ".env").write_text("\n".join(f"{key}={value}" for key, value in overrides.items()))
    assert export_contract(tmp_path, env) == baseline
    assert export_contract(tmp_path, dict(env, **overrides)) == baseline
    assert b"sentinel" not in baseline
    assert json.loads(baseline)["info"]["title"] == "Waste Volume Estimation API"


def test_published_contract_matches_export(tmp_path):
    assert export_contract(tmp_path, dict(os.environ, TMPDIR=str(tmp_path))) == (ROOT / "openapi.json").read_bytes()


def test_published_contract_references_and_operation_ids():
    contract = json.loads((ROOT / "openapi.json").read_text())

    def check_references(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].startswith("#/")
                target = contract
                for part in value["$ref"][2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
            for item in value.values():
                check_references(item)
        elif isinstance(value, list):
            for item in value:
                check_references(item)

    check_references(contract)
    operation_ids = [operation["operationId"] for path in contract["paths"].values()
                     for operation in path.values() if isinstance(operation, dict) and "operationId" in operation]
    assert len(operation_ids) == len(set(operation_ids))
    assert all(path in {"/", "/health", "/files/{path}"} or path.startswith("/api/v1/cameras")
               for path in contract["paths"])
