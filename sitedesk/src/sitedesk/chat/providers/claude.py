"""Anthropic Claude through the official SDK: one Messages API request per turn.

Follows the Claude API guidance: adaptive thinking (the default on current models, so `thinking` is
not sent), thinking and fallback blocks returned unchanged, `refusal` stop reasons handled, and
server-side refusal fallbacks (`fallbacks: "default"`) on the models that support them.
"""

from collections.abc import Sequence
from typing import Any

import anthropic

from ...errors import UpstreamError
from ..config import Endpoint
from ..types import Message, ToolCall, ToolSpec, Turn, Usage

SERVICE = "Claude"
DEFAULT_MAX_TOKENS = (
    16_000  # non-streaming default: leaves room for thinking without hitting HTTP timeouts
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models with server-side refusal fallback. Haiku 5.5 has none, so it must not be sent `fallbacks`.
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)


def _to_api(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            content = (
                m.raw if m.raw else m.text
            )  # raw: the response's content blocks, thinking included
            if content:
                out.append({"role": "assistant", "content": content})
        else:  # every result of one round goes back in a single user message
            blocks = [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content}
                | ({"is_error": True} if r.is_error else {})
                for r in m.results
            ]
            out.append({"role": "user", "content": blocks})
    return out


def _detail(exc: anthropic.APIStatusError) -> str:
    """The vendor's own message: exc.message is the SDK's longer 'Error code: 400 - ...'."""
    error = exc.body.get("error") if isinstance(exc.body, dict) else None
    return (error.get("message") if isinstance(error, dict) else None) or exc.message


class ClaudeProvider:
    def __init__(self, endpoint: Endpoint, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._endpoint = endpoint
        self._client = client or anthropic.AsyncAnthropic(
            api_key=endpoint.api_key, base_url=endpoint.base_url
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(
        self, *, model: str, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Turn:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": self._endpoint.max_output_tokens or DEFAULT_MAX_TOKENS,
            "system": system,
            "messages": _to_api(messages),
        }
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.schema}
                for t in tools
            ]
        try:
            if self._endpoint.refusal_fallbacks and model in FALLBACK_MODELS:
                response = await self._client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = await self._client.messages.create(**params)
        except anthropic.AuthenticationError:
            raise UpstreamError(
                SERVICE, "the API key was rejected: check apiKey in sitedesk.yaml"
            ) from None
        except anthropic.PermissionDeniedError:
            raise UpstreamError(SERVICE, "this API key is not allowed to use that model") from None
        except anthropic.NotFoundError:
            raise UpstreamError(
                SERVICE, f"model {model!r} was not found: check models in sitedesk.yaml"
            ) from None
        except anthropic.RateLimitError:
            raise UpstreamError(SERVICE, "rate limited: try again in a moment") from None
        except anthropic.BadRequestError as exc:
            raise UpstreamError(SERVICE, f"the request was rejected: {_detail(exc)}") from None
        except anthropic.APIStatusError as exc:
            raise UpstreamError(SERVICE, f"HTTP {exc.status_code}") from None
        except anthropic.APIConnectionError:
            raise UpstreamError(SERVICE, "could not reach the Anthropic API") from None

        blocks = response.content
        refused = None
        if response.stop_reason == "refusal":  # a normal 200: check it before using the content
            category = getattr(getattr(response, "stop_details", None), "category", None)
            refused = f"safety filter: {category}" if category else "safety filter"
        usage = response.usage
        return Turn(
            text="".join(b.text for b in blocks if b.type == "text"),
            calls=[ToolCall(b.id, b.name, dict(b.input)) for b in blocks if b.type == "tool_use"],
            usage=Usage(
                (usage.input_tokens or 0)
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
                + (getattr(usage, "cache_read_input_tokens", 0) or 0),
                usage.output_tokens or 0,
            ),
            raw=blocks,
            refused=refused,
            truncated=response.stop_reason == "max_tokens",
        )
