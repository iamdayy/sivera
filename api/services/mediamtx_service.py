import asyncio
import logging
import re
from urllib.parse import urlsplit
from uuid import UUID

import httpx

from api.core.config import Settings, settings


MANAGED_PATH = re.compile(r"^sivera_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class MediaMTXError(Exception):
    def __init__(self, code: str = "stream_unavailable", status_code: int = 503) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


class SignalingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # MediaMTX session URLs contain bearer-like secrets, even without credentials.
        message = record.getMessage()
        if "/whep/" in message:
            record.msg = re.sub(r"(/whep/)[^\s\"?]+", r"\1[redacted]", message)
            record.args = ()
        return True


logging.getLogger("httpx").addFilter(SignalingLogFilter())


def stream_path(camera_id: UUID) -> str:
    return f"sivera_{camera_id}"


def validate_rtsp_source(source: str, config: Settings = settings) -> None:
    try:
        parsed = urlsplit(source)
        port = parsed.port
        host = (parsed.hostname or "").lower()
    except ValueError:
        raise MediaMTXError("invalid_rtsp_source", 422) from None
    if (
        parsed.scheme not in {"rtsp", "rtsps"}
        or not host or parsed.fragment
        or any(ord(char) < 33 for char in source)
        or (port is not None and port == 0)
        or "$" in source
        or "\\" in source
        or not source.startswith(("rtsp://", "rtsps://"))
        or bool(parsed.username) != bool(parsed.password)
        or re.search(r"%(?![0-9a-fA-F]{2})", source)
    ):
        raise MediaMTXError("invalid_rtsp_source", 422)
    allowed = {item.strip().lower() for item in config.MEDIAMTX_ALLOWED_RTSP_HOSTS.split(",") if item.strip()}
    if host not in allowed:
        raise MediaMTXError("rtsp_host_not_allowed", 422)


def desired_path(source: str) -> dict:
    return {
        "source": source,
        "sourceOnDemand": True,
        "sourceOnDemandStartTimeout": "10s",
        "sourceOnDemandCloseAfter": "10s",
        "rtspTransport": "tcp",
    }


class MediaMTXService:
    def __init__(self, config: Settings = settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.STREAM_TIMEOUT, connect=3),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )
        self.api_url = str(httpx.URL(f"http://{config.MEDIAMTX_HOST}:{config.MEDIAMTX_API_PORT}"))
        self.whep_base = str(httpx.URL(f"http://{config.MEDIAMTX_HOST}:{config.MEDIAMTX_WEBRTC_PORT}"))

    async def close(self) -> None:
        await self.client.aclose()

    def check_config(self, read: bool = False) -> httpx.BasicAuth:
        user = self.config.MEDIAMTX_READ_USER if read else self.config.MEDIAMTX_API_USER
        secret = self.config.MEDIAMTX_READ_PASSWORD if read else self.config.MEDIAMTX_API_PASSWORD
        if not self.config.MEDIAMTX_ENABLED or not user or not secret.get_secret_value():
            raise MediaMTXError("stream_service_not_configured")
        return httpx.BasicAuth(user, secret.get_secret_value())

    async def request(
        self, method: str, path: str, *, read: bool = False, retry: bool = False,
        content: bytes | None = None, headers: dict | None = None,
        json: dict | None = None, params: dict | None = None,
    ) -> httpx.Response:
        auth = self.check_config(read)
        attempts = self.config.STREAM_RETRY_ATTEMPTS if retry else 1
        base = self.whep_base if read else self.api_url
        # Cold on-demand offers can wait 10s for the source plus track gathering.
        timeout = self.config.STREAM_TIMEOUT + 15 if read and method == "POST" else self.config.STREAM_TIMEOUT
        for attempt in range(attempts):
            try:
                async with asyncio.timeout(timeout):
                    async with self.client.stream(
                        method, base + path, auth=auth, content=content,
                        headers=headers, json=json, params=params, timeout=timeout,
                    ) as response:
                        limit = self.config.MEDIAMTX_MAX_SDP_BYTES if read else 4 * 1024 * 1024
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > limit:
                                raise MediaMTXError("invalid_stream_response", 502)
                        result = httpx.Response(response.status_code, headers=response.headers, content=bytes(body))
                if result.status_code not in {502, 503, 504} or attempt == attempts - 1:
                    return result
            except (httpx.RequestError, TimeoutError):
                if attempt == attempts - 1:
                    raise MediaMTXError() from None
            await asyncio.sleep(0.1 * (attempt + 1))
        raise MediaMTXError()

    @staticmethod
    def require_success(response: httpx.Response) -> None:
        if not response.is_success:
            raise MediaMTXError()

    @staticmethod
    def object_response(response: httpx.Response) -> dict:
        MediaMTXService.require_success(response)
        try:
            data = response.json()
        except ValueError:
            raise MediaMTXError("invalid_stream_response", 502) from None
        if not isinstance(data, dict):
            raise MediaMTXError("invalid_stream_response", 502)
        return data

    async def get_path_config(self, name: str) -> dict | None:
        response = await self.request("GET", f"/v3/config/paths/get/{name}", retry=True)
        return None if response.status_code == 404 else self.object_response(response)

    async def list_path_configs(self) -> list[dict]:
        items = []
        page = 0
        while True:
            response = await self.request("GET", "/v3/config/paths/list", retry=True, params={"page": page, "itemsPerPage": 100})
            data = self.object_response(response)
            if not isinstance(data.get("items"), list) or not isinstance(data.get("pageCount"), int):
                raise MediaMTXError("invalid_stream_response", 502)
            items.extend(data["items"])
            page += 1
            if page >= data["pageCount"]:
                return items
            if page > 1000:
                raise MediaMTXError("invalid_stream_response", 502)

    async def ensure_path(self, name: str, source: str, current: dict | None) -> None:
        validate_rtsp_source(source, self.config)
        wanted = desired_path(source)
        if current is not None and all(current.get(key) == value for key, value in wanted.items()):
            return
        action = "add" if current is None else "patch"
        response = await self.request("POST" if current is None else "PATCH", f"/v3/config/paths/{action}/{name}", json=wanted)
        self.require_success(response)

    async def delete_path(self, name: str) -> None:
        response = await self.request("DELETE", f"/v3/config/paths/delete/{name}", retry=True)
        if response.status_code != 404:
            self.require_success(response)

    async def runtime_path(self, name: str) -> dict | None:
        response = await self.request("GET", f"/v3/paths/get/{name}", retry=True)
        return None if response.status_code == 404 else self.object_response(response)

    async def signaling(self, method: str, camera_id: UUID, session_id: UUID | None = None, **kwargs) -> httpx.Response:
        path = f"/{stream_path(camera_id)}/whep"
        if session_id is not None:
            path += f"/{session_id}"
        response = await self.request(method, path, read=True, **kwargs)
        expected = {"OPTIONS": {204}, "POST": {201}, "PATCH": {200, 204}, "DELETE": {200, 204, 404}}
        if response.status_code not in expected[method]:
            if session_id is not None and response.status_code == 404:
                raise MediaMTXError("stream_session_not_found", 404)
            if method in {"POST", "PATCH"} and response.status_code in {400, 406}:
                raise MediaMTXError("invalid_sdp_or_unsupported_codec", 400)
            raise MediaMTXError()
        if method == "POST" and response.headers.get("content-type", "").split(";", 1)[0] != "application/sdp":
            raise MediaMTXError("invalid_stream_response", 502)
        return response

    def parse_session_location(self, location: str, camera_id: UUID) -> UUID:
        try:
            parsed = urlsplit(location)
        except ValueError:
            raise MediaMTXError("invalid_stream_response", 502) from None
        expected = urlsplit(self.whep_base)
        prefix = f"/{stream_path(camera_id)}/whep/"
        if (
            (parsed.netloc and (parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc))
            or parsed.query or parsed.fragment or not parsed.path.startswith(prefix)
        ):
            raise MediaMTXError("invalid_stream_response", 502)
        try:
            return UUID(parsed.path[len(prefix):])
        except ValueError:
            raise MediaMTXError("invalid_stream_response", 502) from None
