"""API-owned scheduling and bounded computer-vision execution."""
from api.runtime.manager import EventHub, LocalRuntime, admit_job, notify_runtime, require_runtime

event_hub = EventHub(0)  # placeholder; replaced by main.py with configured max_clients

__all__ = ["event_hub", "EventHub", "LocalRuntime", "admit_job", "notify_runtime", "require_runtime"]
