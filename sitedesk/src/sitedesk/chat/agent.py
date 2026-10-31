"""The chat agent: one question in, tool calls through the MCP server, one grounded answer out.

The loop is vendor-neutral: it talks to a Provider (see providers/) and runs tools in-process on the
same MCPServer that Claude Code or LibreChat reach over MCP, so the chat box and any external MCP
client always see exactly the same tools, descriptions and error messages.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ..mcp_server import INSTRUCTIONS
from .types import Message, Provider, ToolCall, ToolResult, ToolSpec, Usage

log = logging.getLogger(__name__)

TOOL_TIMEOUT_S = 120.0
MAX_RESULT_CHARS = 24_000  # a tool result is sent to the model whole: bound what one call can cost
MAX_ERROR_CHARS = 500

GUIDE = """\
You are SiteDesk's assistant. You help engineers screen sites in Vietnam, mainly for
contaminated-land work, from public data.

- Get every fact about weather, soil, terrain, nearby water, place names and dioxin hotspots
  from the tools, never from memory.
- The tools take WGS84 coordinates. If the user names a place and no point is selected on the map,
  ask for coordinates instead of guessing them.
- Use the cheapest tool that answers the question. get_nearby_water is slow (it can take about a
  minute), so call it only when water proximity matters. get_site_profile is for "tell me about this
  site".
- Be brief: the answer first, then the key numbers with their units. Plain text and short lists;
  no tables unless asked.
"""


def build_system_prompt(point: tuple[float, float] | None, extra: str = "") -> str:
    parts = [GUIDE, INSTRUCTIONS]
    if point is not None:
        parts.append(
            f"The user has selected a point on the map: lat {point[0]}, lon {point[1]}. "
            "Use it when they say 'here', 'this site' or 'this location'."
        )
    if extra.strip():
        parts.append(extra.strip())
    return "\n\n".join(parts)


def _clip(text: str, limit: int) -> str:
    return (
        text
        if len(text) <= limit
        else text[:limit] + f"... [cut: {len(text) - limit} more characters]"
    )


class ToolBox:
    """The SiteDesk MCP server's tools, called in-process."""

    def __init__(self, server: MCPServer) -> None:
        self._server = server

    async def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(t.name, t.description or "", t.input_schema)
            for t in await self._server.list_tools()
        ]

    async def call(self, call: ToolCall) -> ToolResult:
        try:
            result = await asyncio.wait_for(
                self._server.call_tool(call.name, call.args), TOOL_TIMEOUT_S
            )
        except ToolError as exc:
            # unknown tool, bad arguments, upstream failure: the model reads it and self-corrects
            return ToolResult(call.id, call.name, _clip(str(exc), MAX_ERROR_CHARS), is_error=True)
        except TimeoutError:
            return ToolResult(
                call.id, call.name, f"{call.name} timed out after {TOOL_TIMEOUT_S:.0f} s", True
            )
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            text = json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
        else:
            text = "".join(getattr(block, "text", "") for block in result.content)
        return ToolResult(call.id, call.name, _clip(text, MAX_RESULT_CHARS), bool(result.is_error))


async def _timed(box: ToolBox, call: ToolCall) -> tuple[ToolResult, int]:
    start = time.monotonic()
    result = await box.call(call)
    return result, round((time.monotonic() - start) * 1000)


async def run_chat(
    *,
    provider: Provider,
    model: str,
    toolbox: ToolBox,
    message: str,
    history: Sequence[tuple[str, str]] = (),
    point: tuple[float, float] | None = None,
    max_rounds: int = 6,
    extra_prompt: str = "",
) -> AsyncIterator[dict[str, Any]]:
    """Run one question to an answer, yielding progress events for the UI.

    Events: tool_call, tool_result, then answer | error, then done (last, unless an error stops it).
    Provider failures raise UpstreamError; the caller turns that into an error event.
    """
    system = build_system_prompt(point, extra_prompt)
    tools = await toolbox.specs()
    messages = [Message(role=r, text=t) for r, t in history if t.strip()]  # type: ignore[arg-type]
    messages.append(Message(role="user", text=message))
    usage, n_calls, started = Usage(), 0, time.monotonic()

    def done(rounds: int) -> dict[str, Any]:
        return {
            "type": "done",
            "rounds": rounds,
            "tool_calls": n_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "seconds": round(time.monotonic() - started, 1),
        }

    for round_no in range(1, max_rounds + 1):
        turn = await provider.complete(model=model, system=system, messages=messages, tools=tools)
        usage += turn.usage
        if turn.refused:
            yield {
                "type": "error",
                "message": f"The model declined to answer this ({turn.refused}).",
            }
            return
        if not turn.calls:
            text = turn.text.strip() or "(The model returned no text.)"
            if turn.truncated:
                text += "\n\n[The answer was cut off by the output limit.]"
            yield {"type": "answer", "text": text}
            yield done(round_no)
            return

        messages.append(Message(role="assistant", text=turn.text, calls=turn.calls, raw=turn.raw))
        n_calls += len(turn.calls)
        for call in turn.calls:
            yield {"type": "tool_call", "id": call.id, "name": call.name, "args": call.args}
        outcomes = await asyncio.gather(
            *(_timed(toolbox, c) for c in turn.calls)
        )  # parallel calls run together
        for result, ms in outcomes:
            yield {
                "type": "tool_result",
                "id": result.call_id,
                "name": result.name,
                "ok": not result.is_error,
                "ms": ms,
            }
        messages.append(Message(role="tool", results=[r for r, _ in outcomes]))

    yield {
        "type": "error",
        "message": f"Stopped after {max_rounds} rounds of tool calls without an answer.",
    }
