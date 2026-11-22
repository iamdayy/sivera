"""Tests for public-endpoint rate limiting and SSE payload bounds."""
import asyncio
import json
from uuid import uuid4

from api.core.rate_limit import SlidingWindowRateLimiter
from api.runtime.manager import EventHub


def test_rate_limiter_allows_up_to_limit_then_blocks():
    now = [1000.0]
    limiter = SlidingWindowRateLimiter(3, clock=lambda: now[0])
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False
    now[0] += 61  # window slides
    assert limiter.allow("k") is True


def test_rate_limiter_keys_are_independent():
    limiter = SlidingWindowRateLimiter(1)
    assert limiter.allow("a") is True
    assert limiter.allow("b") is True
    assert limiter.allow("a") is False


def test_event_hub_trims_oversized_payload():
    hub = EventHub(max_clients=10, max_event_bytes=256)
    camera = uuid4()
    queue = hub.subscribe(camera)
    event = {"frame_jpeg_base64": "x" * 4096, "segmentation": {"mask_png_base64": "y" * 4096, "has_mask": True}, "volume_liters": 12.5}
    asyncio.run(hub.publish(camera, event))
    payload = json.loads(queue.get_nowait())
    assert payload["payload_truncated"] is True
    assert "frame_jpeg_base64" not in payload
    assert "mask_png_base64" not in payload["segmentation"]
    assert payload["segmentation"]["has_mask"] is True
    assert payload["volume_liters"] == 12.5


def test_event_hub_keeps_small_payload_intact():
    hub = EventHub(max_clients=10, max_event_bytes=65536)
    camera = uuid4()
    queue = hub.subscribe(camera)
    asyncio.run(hub.publish(camera, {"volume_liters": 1.0}))
    payload = json.loads(queue.get_nowait())
    assert payload == {"volume_liters": 1.0}
