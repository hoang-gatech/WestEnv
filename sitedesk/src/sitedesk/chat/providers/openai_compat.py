"""Any OpenAI-compatible Chat Completions API: OpenAI, OpenRouter, Groq, Mistral, local servers."""

import json
from collections.abc import Sequence
from typing import Any

import openai

from ...errors import UpstreamError
from ..config import Endpoint
from ..types import Message, ToolCall, ToolSpec, Turn, Usage

SERVICE = "OpenAI-compatible API"


def _to_api(system: str, messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            if m.raw is not None:
                out.append(m.raw)
            elif m.text:
                out.append({"role": "assistant", "content": m.text})
        else:
            out.extend(
                {"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in m.results
            )
    return out


def _arguments(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _detail(exc: openai.APIStatusError) -> str:
    """The server's own error message (exc.message is the SDK's longer 'Error code: 400 - ...')."""
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get(
        "error", body
    )  # the error object itself, or nested under "error" on some servers
    return (error.get("message") if isinstance(error, dict) else None) or exc.message


class OpenAICompatProvider:
    def __init__(self, endpoint: Endpoint, client: openai.AsyncOpenAI | None = None) -> None:
        self._endpoint = endpoint
        self._client = client or openai.AsyncOpenAI(
            api_key=endpoint.api_key, base_url=endpoint.base_url
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(
        self, *, model: str, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Turn:
        params: dict[str, Any] = {"model": model, "messages": _to_api(system, messages)}
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.schema,
                    },
                }
                for t in tools
            ]
        if self._endpoint.max_output_tokens:
            params["max_tokens"] = self._endpoint.max_output_tokens
        try:
            response = await self._client.chat.completions.create(**params)
        except openai.AuthenticationError:
            raise UpstreamError(
                SERVICE, "the API key was rejected: check apiKey in sitedesk.yaml"
            ) from None
        except openai.PermissionDeniedError:
            raise UpstreamError(SERVICE, "this API key is not allowed to use that model") from None
        except openai.NotFoundError:
            raise UpstreamError(
                SERVICE, f"model {model!r} was not found: check models in sitedesk.yaml"
            ) from None
        except openai.RateLimitError:
            raise UpstreamError(SERVICE, "rate limited or out of quota: try again later") from None
        except openai.BadRequestError as exc:
            raise UpstreamError(SERVICE, f"the request was rejected: {_detail(exc)}") from None
        except openai.APIStatusError as exc:
            raise UpstreamError(SERVICE, f"HTTP {exc.status_code}") from None
        except openai.APIConnectionError:
            raise UpstreamError(
                SERVICE, "could not reach the API (check baseURL in sitedesk.yaml)"
            ) from None

        if not response.choices:
            raise UpstreamError(SERVICE, "the API returned no choices")
        choice = response.choices[0]
        message = choice.message
        tool_calls = [tc for tc in (message.tool_calls or []) if tc.type == "function"]
        calls = [
            ToolCall(tc.id, tc.function.name, _arguments(tc.function.arguments))
            for tc in tool_calls
        ]
        raw: dict[str, Any] = {"role": "assistant", "content": message.content}
        if tool_calls:  # sent back as received: the next request must carry the same calls
            raw["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        usage = response.usage
        return Turn(
            text=message.content or "",
            calls=calls,
            usage=Usage(
                getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0
            ),
            raw=raw,
            refused="content filter" if choice.finish_reason == "content_filter" else None,
            truncated=choice.finish_reason == "length",
        )
