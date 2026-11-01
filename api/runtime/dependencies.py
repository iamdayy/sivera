from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from api.runtime.manager import LocalRuntime


def require_runtime(request: Request) -> "LocalRuntime":
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.status != "running":
        raise HTTPException(status_code=503, detail={"code": "runtime_unavailable"}, headers={"Retry-After": "5"})
    return runtime


def notify_runtime(request: Request | None = None) -> None:
    if request is not None:
        runtime = getattr(request.app.state, "runtime", None)
    else:
        runtime = None
    if runtime is not None:
        runtime.notify()
