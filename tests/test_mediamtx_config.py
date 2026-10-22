from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def services() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]


def environment(service: dict) -> dict[str, str]:
    return dict(item.split("=", 1) for item in service["environment"])


@pytest.mark.parametrize(
    ("protocol", "port"),
    [("API", 9997), ("RTSP", 8554), ("RTMP", 1935), ("HLS", 8888), ("WEBRTC", 8889)],
)
def test_mediamtx_listener_environment(services: dict, protocol: str, port: int) -> None:
    env = environment(services["mediamtx"])
    assert env[f"MTX_{protocol}"] == "true"
    assert env[f"MTX_{protocol}ADDRESS"] == f":{port}"
    assert f"MTX_{protocol}_ENABLED" not in env
    assert f"MTX_{protocol}_ADDRESS" not in env


@pytest.mark.parametrize("service", ["api", "worker"])
@pytest.mark.parametrize("protocol", ["HLS", "WEBRTC"])
def test_backend_ports_match_listeners(services: dict, service: str, protocol: str) -> None:
    listener = environment(services["mediamtx"])[f"MTX_{protocol}ADDRESS"]
    assert environment(services[service])[f"MEDIAMTX_{protocol}_PORT"] == listener.lstrip(":")


@pytest.mark.parametrize("protocol", ["HLS", "WEBRTC"])
def test_example_ports_match_listeners(services: dict, protocol: str) -> None:
    example = dict(
        line.split("=", 1)
        for line in (ROOT / ".env.example").read_text().splitlines()
        if line and not line.startswith("#")
    )
    listener = environment(services["mediamtx"])[f"MTX_{protocol}ADDRESS"]
    assert example[f"MEDIAMTX_{protocol}_PORT"] == listener.lstrip(":")


def test_mediamtx_healthcheck_uses_get(services: dict) -> None:
    command = services["mediamtx"]["healthcheck"]["test"]
    assert command[:2] == ["CMD", "wget"]
    assert "--spider" not in command
    assert "-O" in command
    assert "/dev/null" in command
    assert command[-1] == "http://127.0.0.1:9997/v3/paths/list"


def test_mediamtx_control_port_is_local_only(services: dict) -> None:
    bindings = [port for port in services["mediamtx"]["ports"] if port.endswith(":9997")]
    assert bindings == ["127.0.0.1:9997:9997"]


def test_private_signaling_and_public_ice(services: dict) -> None:
    ports = services["mediamtx"]["ports"]
    assert "127.0.0.1:8889:8889" in ports
    assert "8189:8189/udp" in ports
    env = environment(services["mediamtx"])
    assert env["MTX_WEBRTCIPSFROMINTERFACES"] == "false"
    assert "MTX_WEBRTCIPFROMINTERFACES" not in env


def test_authenticated_read_and_control_users(services: dict) -> None:
    service = services["mediamtx"]
    env = environment(service)
    assert ":?" in env["MTX_AUTHINTERNALUSERS_1_PASS"]
    assert ":?" in env["MTX_AUTHINTERNALUSERS_2_PASS"]
    config = yaml.safe_load((ROOT / "deploy/mediamtx.yml").read_text())
    assert config["paths"] == {}
    users = config["authInternalUsers"]
    assert users[0]["ips"] == ["127.0.0.1", "::1"]
    assert users[1]["permissions"] == [{"action": "api"}]
    assert users[2]["permissions"][0]["action"] == "read"
    assert "./deploy/mediamtx.yml:/mediamtx.yml:ro" in service["volumes"]


def test_shared_artifacts_and_separate_workers(services: dict) -> None:
    for name in ("api", "worker", "capture-worker"):
        assert "artifacts:/app/storage" in services[name]["volumes"]
        assert environment(services[name])["STORAGE_LOCAL_ROOT"] == "/app/storage"
    assert "-Q prediction,celery" in services["worker"]["command"]
    assert "-Q capture" in services["capture-worker"]["command"]
    assert "--concurrency=1" in services["capture-worker"]["command"]
    assert "ffmpeg" in (ROOT / "Dockerfile").read_text().split("# Runtime stage")[1]
