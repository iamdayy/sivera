"""API-owned scheduling and bounded computer-vision execution."""
from api.runtime.events import EventHub

event_hub = EventHub(0)  # placeholder; replaced by main.py with configured max_clients

__all__ = ["event_hub", "EventHub"]
