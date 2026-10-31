"""Shared async HTTP client: retries transient failures, normalises errors to UpstreamError."""

import asyncio
import logging
from typing import Any

import httpx

from .config import settings
from .errors import UpstreamError

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_S = 1.0  # delay before the first retry, doubled each time; tests set this to 0
MAX_RETRY_AFTER_S = 10.0

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return min(float(resp.headers["Retry-After"]), MAX_RETRY_AFTER_S)
    except (KeyError, ValueError):
        return None


async def request_json(
    service: str,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    timeout: float = 20.0,  # noqa: ASYNC109 - enforced per request by httpx, not by this coroutine
    retries: int = 2,
) -> Any:
    """Call an upstream API and return its decoded JSON, or raise UpstreamError.

    Timeouts, connection errors, 429 and 5xx are retried with exponential backoff; any other
    4xx fails immediately because retrying will not change the answer.
    """
    delay = BACKOFF_S
    for attempt in range(retries + 1):
        retry_after = None
        try:
            resp = await client().request(method, url, params=params, data=data, timeout=timeout)
        except httpx.TransportError as exc:  # includes timeouts, which often have an empty message
            failure = type(exc).__name__ + (f": {exc}" if str(exc) else "")
        else:
            if resp.status_code < 400:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise UpstreamError(service, "response was not valid JSON") from exc
            if resp.status_code not in RETRY_STATUS:
                raise UpstreamError(service, f"HTTP {resp.status_code}: {resp.text[:200]!r}")
            failure = f"HTTP {resp.status_code}"
            retry_after = _retry_after(resp)
        if attempt == retries:
            raise UpstreamError(service, f"{failure} (gave up after {retries + 1} attempts)")
        log.warning("%s attempt %d/%d failed: %s", service, attempt + 1, retries + 1, failure)
        await asyncio.sleep(retry_after if retry_after is not None else delay)
        delay *= 2
    raise AssertionError("unreachable")


async def get_json(service: str, url: str, **kwargs: Any) -> Any:
    return await request_json(service, "GET", url, **kwargs)


async def post_form_json(service: str, url: str, data: dict[str, Any], **kwargs: Any) -> Any:
    return await request_json(service, "POST", url, data=data, **kwargs)
