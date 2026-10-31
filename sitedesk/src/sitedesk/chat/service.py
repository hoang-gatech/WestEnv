"""The chat service behind /api/chat: config lookup, provider cache, one streamed question."""

import hashlib
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..errors import UpstreamError
from ..mcp_server import mcp
from .agent import ToolBox, run_chat
from .config import DEFAULT_PATH, ChatConfig, ChatConfigError, ChatSettings, Endpoint, load_config
from .providers import create_provider
from .types import Provider

log = logging.getLogger(__name__)

MAX_CONCURRENT = 3  # questions answered at once: each can run for minutes and costs money


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class Point(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    endpoint: str | None = None
    model: str | None = None
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)
    point: Point | None = None


class ChatError(Exception):
    """A request that cannot be served. `status` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Prepared:
    endpoint: Endpoint
    model: str
    provider: Provider
    settings: ChatSettings


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


class ChatService:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(os.environ.get("SITEDESK_CONFIG") or DEFAULT_PATH)
        self._loaded: tuple[tuple[int, int], ChatConfig | ChatConfigError] | None = None
        self._providers: dict[tuple[str, str, str, str], Provider] = {}
        self._toolbox = ToolBox(mcp)
        self._active = 0

    def config(self) -> ChatConfig:
        """The current config, re-read when sitedesk.yaml or its .env changes (no restart)."""
        stamp = (_mtime(self._path), _mtime(self._path.with_name(".env")))
        if self._loaded is None or self._loaded[0] != stamp:
            try:
                self._loaded = (stamp, load_config(self._path))
            except ChatConfigError as exc:
                self._loaded = (stamp, exc)
        value = self._loaded[1]
        if isinstance(value, ChatConfigError):
            raise value
        return value

    def public_config(self) -> dict[str, Any]:
        """What the browser may know: names and models, never keys."""
        try:
            cfg = self.config()
        except ChatConfigError as exc:
            return {"enabled": False, "problem": str(exc), "default": None, "endpoints": []}
        ready = [e for e in cfg.endpoints if e.has_key]
        default = (
            cfg.chat.default_endpoint
            if any(e.name == cfg.chat.default_endpoint for e in ready)
            else None
        )
        return {
            "enabled": bool(ready),
            "problem": None
            if ready
            else "No endpoint has an API key yet: set the keys sitedesk.yaml refers to.",
            "default": default or (ready[0].name if ready else None),
            "endpoints": [
                {"name": e.name, "provider": e.provider, "models": e.models, "ready": e.has_key}
                for e in cfg.endpoints
            ],
        }

    def prepare(self, req: ChatRequest) -> Prepared:
        """What can be refused before the stream starts, so it gets a proper HTTP status."""
        try:
            cfg = self.config()
        except ChatConfigError as exc:
            raise ChatError(503, str(exc)) from None
        wanted = req.endpoint or self.public_config()["default"]
        endpoint = next((e for e in cfg.endpoints if e.name == wanted), None)
        if endpoint is None:
            raise ChatError(400 if req.endpoint else 503, f"unknown or missing endpoint {wanted!r}")
        if not endpoint.has_key:
            raise ChatError(
                503,
                f"Endpoint {endpoint.name!r} has no API key: set it in sitedesk.yaml or the env.",
            )
        model = req.model or endpoint.models[0]
        if (
            model not in endpoint.models
        ):  # an allow-list: the browser cannot pick an arbitrary, costlier model
            raise ChatError(
                400, f"model {model!r} is not configured for endpoint {endpoint.name!r}"
            )
        if self._active >= MAX_CONCURRENT:
            raise ChatError(
                429, "Too many questions are being answered at once: try again in a moment."
            )
        try:
            provider = self._provider(endpoint)
        except ChatConfigError as exc:
            raise ChatError(503, str(exc)) from None
        return Prepared(endpoint, model, provider, cfg.chat)

    def _provider(self, endpoint: Endpoint) -> Provider:
        key = (
            endpoint.name,
            endpoint.provider,
            endpoint.base_url or "",
            hashlib.sha256(endpoint.api_key.encode()).hexdigest(),
        )
        if (
            key not in self._providers
        ):  # one client (and connection pool) per endpoint, reused across questions
            self._providers[key] = create_provider(endpoint)
        return self._providers[key]

    async def stream(self, prepared: Prepared, req: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        self._active += 1
        try:
            async for event in run_chat(
                provider=prepared.provider,
                model=prepared.model,
                toolbox=self._toolbox,
                message=req.message,
                history=[(h.role, h.content) for h in req.history],
                point=(req.point.lat, req.point.lon) if req.point else None,
                max_rounds=prepared.settings.max_tool_rounds,
                extra_prompt=prepared.settings.system_prompt,
            ):
                if event["type"] == "done":  # metrics only: never the question or the answer
                    log.info(
                        "chat endpoint=%s model=%s rounds=%d tools=%d in=%d out=%d %.1fs",
                        prepared.endpoint.name,
                        prepared.model,
                        event["rounds"],
                        event["tool_calls"],
                        event["input_tokens"],
                        event["output_tokens"],
                        event["seconds"],
                    )
                yield event
        except UpstreamError as exc:
            yield {"type": "error", "message": str(exc)}
        except Exception:
            log.exception("chat failed")
            yield {"type": "error", "message": "Something went wrong on the server: see its log."}
        finally:
            self._active -= 1

    async def aclose(self) -> None:
        providers, self._providers = list(self._providers.values()), {}
        for provider in providers:
            await provider.aclose()


_service: ChatService | None = None


def chat_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService()
    return _service


async def shutdown_chat() -> None:
    if _service is not None:
        await _service.aclose()
