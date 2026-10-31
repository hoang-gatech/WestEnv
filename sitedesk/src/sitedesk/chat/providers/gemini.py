"""Google Gemini through the official google-genai SDK: one generateContent request per turn."""

import json
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import errors, types

from ...errors import UpstreamError
from ..config import Endpoint
from ..types import Message, ToolCall, ToolSpec, Turn, Usage

SERVICE = "Gemini"
_SYNTHETIC = "g-"  # prefix of ids made up here for function calls the API gave none


def _response_payload(content: str, is_error: bool) -> dict[str, Any]:
    if is_error:
        return {"error": content}
    try:
        return {"result": json.loads(content)}
    except ValueError:
        return {"result": content}


def _to_contents(messages: Sequence[Message]) -> list[types.Content]:
    out: list[types.Content] = []
    for m in messages:
        if m.role == "user":
            out.append(types.Content(role="user", parts=[types.Part(text=m.text)]))
        elif m.role == "assistant":
            if m.raw is not None:
                out.append(
                    m.raw
                )  # the model's own Content: Gemini 3 needs its thought signatures back
            elif m.text:
                out.append(types.Content(role="model", parts=[types.Part(text=m.text)]))
        else:  # all results of one round in one turn, in the order the calls came
            parts = [
                types.Part(
                    function_response=types.FunctionResponse(
                        id=None if r.call_id.startswith(_SYNTHETIC) else r.call_id,
                        name=r.name,
                        response=_response_payload(r.content, r.is_error),
                    )
                )
                for r in m.results
            ]
            out.append(types.Content(role="user", parts=parts))
    return out


def _api_error(exc: errors.APIError, model: str) -> UpstreamError:
    message = (exc.message or "")[:300]
    if exc.code == 400 and "api key" in message.lower():
        return UpstreamError(SERVICE, "the API key was rejected: check apiKey in sitedesk.yaml")
    if exc.code in (401, 403):
        return UpstreamError(
            SERVICE, "the API key was rejected or is not allowed to use this model"
        )
    if exc.code == 404:
        return UpstreamError(
            SERVICE, f"model {model!r} was not found: check models in sitedesk.yaml"
        )
    if exc.code == 429:
        return UpstreamError(
            SERVICE, "rate limit or quota exceeded (the free tier's limits are low)"
        )
    return UpstreamError(SERVICE, f"HTTP {exc.code}: {message}")


class GeminiProvider:
    def __init__(self, endpoint: Endpoint, client: genai.Client | None = None) -> None:
        self._endpoint = endpoint
        http = types.HttpOptions(base_url=endpoint.base_url) if endpoint.base_url else None
        self._client = client or genai.Client(api_key=endpoint.api_key, http_options=http)

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def complete(
        self, *, model: str, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Turn:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[
                types.Tool(
                    function_declarations=[
                        # the tool's JSON Schema as it is, as the SDK's own MCP support does
                        types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=t.schema
                        )
                        for t in tools
                    ]
                )
            ]
            if tools
            else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),  # our loop runs the tools
            max_output_tokens=self._endpoint.max_output_tokens,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=model, contents=_to_contents(messages), config=config
            )
        except errors.APIError as exc:
            raise _api_error(exc, model) from None

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            feedback = response.prompt_feedback
            reason = (
                feedback.block_reason.name
                if feedback and feedback.block_reason
                else "no candidate returned"
            )
            return Turn(refused=f"blocked: {reason}", usage=_usage(response))
        finish = candidate.finish_reason.name if candidate.finish_reason else ""
        if finish == "MALFORMED_FUNCTION_CALL":
            raise UpstreamError(SERVICE, "the model produced a malformed tool call: try again")

        parts = candidate.content.parts or []
        calls = [
            ToolCall(
                p.function_call.id or f"{_SYNTHETIC}{i}",
                p.function_call.name or "",
                dict(p.function_call.args or {}),
            )
            for i, p in enumerate(p for p in parts if p.function_call)
        ]
        refused = (
            f"blocked: {finish}"
            if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
            else None
        )
        return Turn(
            text="".join(p.text for p in parts if p.text and not p.thought),
            calls=calls,
            usage=_usage(response),
            raw=candidate.content,
            refused=refused,
            truncated=finish == "MAX_TOKENS",
        )


def _usage(response: types.GenerateContentResponse) -> Usage:
    meta = response.usage_metadata
    if meta is None:
        return Usage()
    return Usage(
        meta.prompt_token_count or 0,
        (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0),
    )
