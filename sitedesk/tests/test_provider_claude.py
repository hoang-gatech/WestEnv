"""The Claude adapter on the real anthropic SDK, against a fake Messages API."""

import anthropic
import pytest

from sitedesk.chat.config import Endpoint
from sitedesk.chat.providers.claude import FALLBACK_BETA, ClaudeProvider
from sitedesk.chat.types import Message, ToolCall, ToolResult, ToolSpec
from sitedesk.errors import UpstreamError

SOIL_SCHEMA = {
    "type": "object",
    "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
    "required": ["lat", "lon"],
}
TOOLS = [ToolSpec("get_soil", "Soil at a point.", SOIL_SCHEMA)]
HI = [Message(role="user", text="hi")]


def message(content, stop_reason="end_turn", usage=None, **extra):
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 100, "output_tokens": 20},
        **extra,
    }


def text(t):
    return {"type": "text", "text": t}


def tool_use(id="toolu_1", name="get_soil", **args):
    return {
        "type": "tool_use",
        "id": id,
        "name": name,
        "input": args or {"lat": 10.978, "lon": 106.815},
    }


@pytest.fixture
async def make(vendor):
    made: list[ClaudeProvider] = []

    def build(model="claude-opus-5-5", **endpoint_kwargs):
        endpoint = Endpoint(
            name="claude",
            provider="anthropic",
            apiKey="sk-ant-test",
            models=[model],
            **endpoint_kwargs,
        )
        client = anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url=vendor.url, max_retries=0)
        made.append(ClaudeProvider(endpoint, client=client))
        return made[-1]

    yield build
    for provider in made:
        await provider.aclose()


async def test_request_on_the_wire_for_a_model_with_refusal_fallbacks(make, vendor):
    vendor.reply(message([text("Hello.")]))
    turn = await make().complete(
        model="claude-opus-5-5", system="Be brief.", messages=HI, tools=TOOLS
    )
    sent = vendor.last
    assert sent["path"] == "/v1/messages"
    assert sent["headers"]["authorization"] == "Bearer sk-ant-test"  # how SDK 2.x sends the key
    assert FALLBACK_BETA in sent["headers"]["anthropic-beta"]  # the beta is opted into by header
    body = sent["json"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 16000
    assert body["system"] == "Be brief."
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["tools"] == [
        {"name": "get_soil", "description": "Soil at a point.", "input_schema": SOIL_SCHEMA}
    ]
    assert body["fallbacks"] == "default"
    assert "thinking" not in body  # adaptive thinking is the default: nothing to send
    assert (turn.text, turn.calls, turn.refused, turn.truncated) == ("Hello.", [], None, False)
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (100, 20)


@pytest.mark.parametrize(
    ("model", "endpoint_kwargs"),
    [
        ("claude-haiku-5-5", {}),  # no server-side fallback exists for Haiku 5.5
        ("claude-opus-5-5", {"refusalFallbacks": False}),  # switched off in the config
        ("claude-opus-4-8", {}),  # not one of the models that support it
    ],
)
async def test_no_fallbacks_where_they_do_not_apply(make, vendor, model, endpoint_kwargs):
    vendor.reply(message([text("ok")]))
    await make(model, **endpoint_kwargs).complete(model=model, system="s", messages=HI, tools=[])
    assert vendor.last["path"] == "/v1/messages"
    assert "fallbacks" not in vendor.last["json"] and "tools" not in vendor.last["json"]
    assert "anthropic-beta" not in vendor.last["headers"]


async def test_tool_call_round_trip_returns_the_assistant_blocks_untouched(make, vendor):
    provider = make()
    vendor.reply(message([text("Checking."), tool_use()], stop_reason="tool_use"))
    turn = await provider.complete(model="claude-opus-5-5", system="s", messages=HI, tools=TOOLS)
    assert turn.text == "Checking."
    assert turn.calls == [ToolCall("toolu_1", "get_soil", {"lat": 10.978, "lon": 106.815})]

    vendor.reply(message([text("Acrisols.")]))
    history = [
        *HI,
        Message(role="assistant", text=turn.text, calls=turn.calls, raw=turn.raw),
        Message(role="tool", results=[ToolResult("toolu_1", "get_soil", '{"wrb":"Acrisols"}')]),
    ]
    await provider.complete(model="claude-opus-5-5", system="s", messages=history, tools=TOOLS)
    assistant, results = vendor.last["json"]["messages"][1:]
    assert assistant["role"] == "assistant"
    echoed = [b for b in assistant["content"] if b["type"] == "tool_use"]
    assert (
        echoed
        and echoed[0]["id"] == "toolu_1"
        and echoed[0]["input"] == {"lat": 10.978, "lon": 106.815}
    )
    assert results == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"wrb":"Acrisols"}'}
        ],
    }


async def test_every_result_of_a_round_goes_back_in_one_message_and_errors_are_flagged(
    make, vendor
):
    vendor.reply(message([text("ok")]))
    history = [
        *HI,
        Message(
            role="assistant",
            text="",
            calls=[ToolCall("a", "get_soil", {}), ToolCall("b", "get_soil", {})],
            raw=[tool_use("a"), tool_use("b")],
        ),
        Message(
            role="tool",
            results=[
                ToolResult("a", "get_soil", "fine"),
                ToolResult("b", "get_soil", "SoilGrids: HTTP 503", is_error=True),
            ],
        ),
    ]
    await make().complete(model="claude-opus-5-5", system="s", messages=history, tools=TOOLS)
    last = vendor.last["json"]["messages"][-1]
    assert [b["tool_use_id"] for b in last["content"]] == ["a", "b"]
    assert "is_error" not in last["content"][0] and last["content"][1]["is_error"] is True


async def test_plain_text_history_without_raw_blocks_is_sent_as_text(make, vendor):
    vendor.reply(message([text("ok")]))
    history = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="hello"),
        Message(role="assistant", text=""),  # nothing to send: the API rejects empty content
        Message(role="user", text="again"),
    ]
    await make().complete(model="claude-haiku-5-5", system="s", messages=history, tools=[])
    assert vendor.last["json"]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again"},
    ]


async def test_a_refusal_is_reported_with_its_category(make, vendor):
    vendor.reply(
        message(
            [],
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "bio", "explanation": "x"},
        )
    )
    turn = await make().complete(model="claude-opus-5-5", system="s", messages=HI, tools=[])
    assert turn.refused == "safety filter: bio"


async def test_hitting_max_tokens_is_flagged_and_cache_tokens_are_counted(make, vendor):
    vendor.reply(
        message(
            [text("Partial")],
            stop_reason="max_tokens",
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 90,
                "cache_creation_input_tokens": 0,
            },
        )
    )
    turn = await make().complete(model="claude-opus-5-5", system="s", messages=HI, tools=[])
    assert turn.truncated and turn.usage.input_tokens == 100 and turn.usage.output_tokens == 5


async def test_a_configured_output_limit_replaces_the_default(make, vendor):
    vendor.reply(message([text("ok")]))
    await make(maxOutputTokens=2000).complete(
        model="claude-opus-5-5", system="s", messages=HI, tools=[]
    )
    assert vendor.last["json"]["max_tokens"] == 2000


@pytest.mark.parametrize(
    ("status", "kind", "expected"),
    [
        (401, "authentication_error", "API key was rejected"),
        (403, "permission_error", "not allowed to use that model"),
        (404, "not_found_error", "model 'claude-opus-5-5' was not found"),
        (429, "rate_limit_error", "rate limited"),
        (400, "invalid_request_error", "the request was rejected: bad shape"),
        (500, "api_error", "HTTP 500"),
    ],
)
async def test_vendor_errors_become_readable_messages_that_never_contain_the_key(
    make, vendor, status, kind, expected
):
    vendor.reply({"type": "error", "error": {"type": kind, "message": "bad shape"}}, status=status)
    with pytest.raises(UpstreamError, match=expected) as exc:
        await make().complete(model="claude-opus-5-5", system="s", messages=HI, tools=[])
    assert "sk-ant-test" not in str(exc.value) and str(exc.value).startswith("Claude:")


async def test_an_unreachable_server_is_reported(make):
    endpoint = Endpoint(
        name="claude", provider="anthropic", apiKey="sk-ant-test", models=["claude-opus-5-5"]
    )
    dead = anthropic.AsyncAnthropic(
        api_key="sk-ant-test", base_url="http://127.0.0.1:9", max_retries=0
    )
    provider = ClaudeProvider(endpoint, client=dead)
    with pytest.raises(UpstreamError, match="could not reach the Anthropic API"):
        await provider.complete(model="claude-opus-5-5", system="s", messages=HI, tools=[])
    await provider.aclose()
