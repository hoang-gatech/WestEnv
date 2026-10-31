import asyncio
import json
import time

from chat_helpers import BIEN_HOA, ScriptedProvider, call

from sitedesk.chat import agent
from sitedesk.chat.agent import ToolBox, build_system_prompt, run_chat
from sitedesk.chat.types import ToolResult, Turn, Usage
from sitedesk.mcp_server import mcp

ALL_TOOLS = {
    "get_weather",
    "get_soil",
    "get_terrain",
    "get_nearby_water",
    "get_known_hotspots",
    "reverse_geocode",
    "get_site_profile",
}


async def collect(provider, **kwargs):
    kwargs.setdefault("toolbox", ToolBox(mcp))
    kwargs.setdefault("message", "What is near here?")
    return [event async for event in run_chat(provider=provider, model="m", **kwargs)]


async def test_answers_directly_when_no_tool_is_needed():
    provider = ScriptedProvider(Turn(text="  Hello.  ", usage=Usage(120, 8)))
    events = await collect(provider)
    assert [e["type"] for e in events] == ["answer", "done"]
    assert events[0]["text"] == "Hello."
    assert events[1] == {
        **events[1],
        "rounds": 1,
        "tool_calls": 0,
        "input_tokens": 120,
        "output_tokens": 8,
    }
    request = provider.requests[0]
    assert {t.name for t in request["tools"]} == ALL_TOOLS
    assert (
        "never a survey or laboratory result" in request["system"]
    )  # the MCP server's own instructions
    assert "never from memory" in request["system"]


async def test_runs_a_tool_and_sends_the_result_back_to_the_model():
    first = Turn(text="Checking.", calls=[call()], usage=Usage(100, 20), raw={"vendor": "payload"})
    provider = ScriptedProvider(first, Turn(text="Bien Hoa is on site.", usage=Usage(300, 30)))
    events = await collect(provider)
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "answer", "done"]
    assert events[0] == {
        "type": "tool_call",
        "id": "c1",
        "name": "get_known_hotspots",
        "args": BIEN_HOA,
    }
    assert events[1]["ok"] is True and events[1]["ms"] >= 0
    assert events[3] == {
        **events[3],
        "rounds": 2,
        "tool_calls": 1,
        "input_tokens": 400,
        "output_tokens": 50,
    }

    sent = provider.requests[1]["messages"]
    assert [m.role for m in sent] == ["user", "assistant", "tool"]
    assert sent[1].calls == first.calls and sent[1].raw == {"vendor": "payload"}  # echoed untouched
    (result,) = sent[2].results
    assert result.call_id == "c1" and not result.is_error
    assert json.loads(result.content)["nearest"]["id"] == "bien-hoa"  # real tool output, as JSON


async def test_tool_results_are_compact_json():
    result = await ToolBox(mcp).call(call())
    assert "\n" not in result.content and '"nearest":' in result.content  # tokens cost money


async def test_parallel_calls_run_together_and_keep_their_order():
    class SlowBox:
        async def specs(self):
            return []

        async def call(self, c):
            await asyncio.sleep(0.2)
            return ToolResult(c.id, c.name, f"done {c.id}")

    provider = ScriptedProvider(Turn(calls=[call("a"), call("b"), call("c")]), Turn(text="ok"))
    start = time.monotonic()
    events = await collect(provider, toolbox=SlowBox())
    assert time.monotonic() - start < 0.45  # three 0.2 s calls took about 0.2 s
    assert [e["id"] for e in events if e["type"] == "tool_result"] == ["a", "b", "c"]
    assert [r.content for r in provider.requests[1]["messages"][-1].results] == [
        "done a",
        "done b",
        "done c",
    ]


async def test_bad_arguments_go_back_to_the_model_as_an_error_it_can_correct():
    provider = ScriptedProvider(
        Turn(calls=[call(lat=95, lon=106.8)]), Turn(text="Sorry, retrying.")
    )
    events = await collect(provider)
    assert events[1]["type"] == "tool_result" and events[1]["ok"] is False
    (result,) = provider.requests[1]["messages"][-1].results
    assert result.is_error and "validation error" in result.content and "lat" in result.content
    assert events[-2]["type"] == "answer"  # the loop carried on


async def test_an_unknown_tool_is_reported_to_the_model():
    provider = ScriptedProvider(Turn(calls=[call(name="no_such_tool", x=1)]), Turn(text="ok"))
    await collect(provider)
    (result,) = provider.requests[1]["messages"][-1].results
    assert result.is_error and "no_such_tool" in result.content


async def test_a_tool_that_hangs_is_cut_off(monkeypatch):
    class HangingServer:
        async def list_tools(self):
            return []

        async def call_tool(self, name, args):
            await asyncio.sleep(5)

    monkeypatch.setattr(agent, "TOOL_TIMEOUT_S", 0.05)
    result = await ToolBox(HangingServer()).call(call())
    assert result.is_error and "timed out" in result.content


async def test_stops_after_the_round_limit_instead_of_looping_forever():
    provider = ScriptedProvider(Turn(calls=[call()]))  # asks for a tool every single time
    events = await collect(provider, max_rounds=3)
    assert len(provider.requests) == 3
    assert events[-1]["type"] == "error" and "3 rounds" in events[-1]["message"]
    assert not any(e["type"] == "answer" for e in events)


async def test_a_refusal_is_reported_as_an_error():
    provider = ScriptedProvider(Turn(refused="safety filter: bio"))
    events = await collect(provider)
    assert events == [
        {"type": "error", "message": "The model declined to answer this (safety filter: bio)."}
    ]


async def test_a_truncated_answer_says_so():
    events = await collect(ScriptedProvider(Turn(text="Partial", truncated=True)))
    assert "cut off by the output limit" in events[0]["text"]


async def test_an_empty_answer_is_not_shown_as_blank():
    events = await collect(ScriptedProvider(Turn(text="   ")))
    assert events[0]["text"] == "(The model returned no text.)"


async def test_history_and_the_selected_point_reach_the_model():
    provider = ScriptedProvider(Turn(text="ok"))
    await collect(
        provider,
        history=[("user", "hi"), ("assistant", "hello"), ("user", "   ")],
        point=(10.978, 106.815),
        extra_prompt="Answer in Vietnamese.",
    )
    request = provider.requests[0]
    assert [(m.role, m.text) for m in request["messages"]] == [
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "What is near here?"),
    ]  # the blank history entry was dropped
    assert "lat 10.978, lon 106.815" in request["system"]
    assert request["system"].endswith("Answer in Vietnamese.")


def test_no_point_means_no_point_in_the_prompt():
    assert "selected a point" not in build_system_prompt(None)


def test_long_tool_output_is_clipped():
    clipped = agent._clip("x" * 100, 10)
    assert clipped.startswith("x" * 10) and "90 more characters" in clipped
    assert agent._clip("short", 10) == "short"
