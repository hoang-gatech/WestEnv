"""The OpenAI-compatible adapter on the real openai SDK, against a fake API."""

import json
import sys

import openai
import pytest

from sitedesk.chat.config import ChatConfigError, Endpoint
from sitedesk.chat.providers import create_provider
from sitedesk.chat.providers.openai_compat import OpenAICompatProvider
from sitedesk.chat.types import Message, ToolCall, ToolResult, ToolSpec
from sitedesk.errors import UpstreamError

SOIL_SCHEMA = {
    "type": "object",
    "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
    "required": ["lat", "lon"],
}
TOOLS = [ToolSpec("get_soil", "Soil at a point.", SOIL_SCHEMA)]
HI = [Message(role="user", text="hi")]
ARGS = json.dumps({"lat": 10.978, "lon": 106.815})


def completion(message, finish="stop", usage=(100, 20)):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        },
    }


def tool_call(id="call_1", name="get_soil", arguments=ARGS):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.fixture
async def make(vendor):
    made: list[OpenAICompatProvider] = []

    def build(**endpoint_kwargs):
        endpoint = Endpoint(
            name="x",
            provider="openai",
            apiKey="sk-test",
            baseURL=vendor.url + "/v1",
            models=["m"],
            **endpoint_kwargs,
        )
        client = openai.AsyncOpenAI(api_key="sk-test", base_url=vendor.url + "/v1", max_retries=0)
        made.append(OpenAICompatProvider(endpoint, client=client))
        return made[-1]

    yield build
    for provider in made:
        await provider.aclose()


async def test_request_on_the_wire(make, vendor):
    vendor.reply(completion({"content": "Hello."}))
    turn = await make().complete(model="m", system="Be brief.", messages=HI, tools=TOOLS)
    sent = vendor.last
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer sk-test"
    body = sent["json"]
    assert body["model"] == "m"
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_soil",
                "description": "Soil at a point.",
                "parameters": SOIL_SCHEMA,
            },
        }
    ]
    assert "max_tokens" not in body  # no limit configured
    assert (turn.text, turn.calls, turn.refused, turn.truncated) == ("Hello.", [], None, False)
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (100, 20)


async def test_tool_call_round_trip_resends_the_calls_and_answers_each_by_id(make, vendor):
    provider = make()
    vendor.reply(
        completion(
            {"content": None, "tool_calls": [tool_call("call_1"), tool_call("call_2")]},
            finish="tool_calls",
        )
    )
    turn = await provider.complete(model="m", system="s", messages=HI, tools=TOOLS)
    assert [c.id for c in turn.calls] == ["call_1", "call_2"]
    assert turn.calls[0].args == {
        "lat": 10.978,
        "lon": 106.815,
    }  # arguments arrive as a JSON string

    vendor.reply(completion({"content": "Acrisols."}))
    history = [
        *HI,
        Message(role="assistant", calls=turn.calls, raw=turn.raw),
        Message(
            role="tool",
            results=[
                ToolResult("call_1", "get_soil", "one"),
                ToolResult("call_2", "get_soil", "two", is_error=True),
            ],
        ),
    ]
    await provider.complete(model="m", system="s", messages=history, tools=TOOLS)
    _, _, assistant, first, second = vendor.last["json"]["messages"]
    assert assistant["role"] == "assistant" and [c["id"] for c in assistant["tool_calls"]] == [
        "call_1",
        "call_2",
    ]
    assert assistant["tool_calls"][0]["function"] == {"name": "get_soil", "arguments": ARGS}
    assert first == {"role": "tool", "tool_call_id": "call_1", "content": "one"}
    assert second == {"role": "tool", "tool_call_id": "call_2", "content": "two"}


async def test_unparseable_arguments_become_an_empty_call_the_tool_will_reject(make, vendor):
    vendor.reply(
        completion(
            {"content": None, "tool_calls": [tool_call(arguments="{not json")]}, finish="tool_calls"
        )
    )
    turn = await make().complete(model="m", system="s", messages=HI, tools=TOOLS)
    assert turn.calls == [ToolCall("call_1", "get_soil", {})]


async def test_a_configured_output_limit_is_sent(make, vendor):
    vendor.reply(completion({"content": "ok"}))
    await make(maxOutputTokens=300).complete(model="m", system="s", messages=HI, tools=[])
    assert vendor.last["json"]["max_tokens"] == 300 and "tools" not in vendor.last["json"]


@pytest.mark.parametrize(
    ("finish", "attr", "expected"),
    [("length", "truncated", True), ("content_filter", "refused", "content filter")],
)
async def test_finish_reasons(make, vendor, finish, attr, expected):
    vendor.reply(completion({"content": "partial"}, finish=finish))
    assert (
        getattr(await make().complete(model="m", system="s", messages=HI, tools=[]), attr)
        == expected
    )


async def test_a_reply_with_no_choices_is_an_error(make, vendor):
    vendor.reply({**completion({"content": "x"}), "choices": []})
    with pytest.raises(UpstreamError, match="no choices"):
        await make().complete(model="m", system="s", messages=HI, tools=[])


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "API key was rejected"),
        (403, "not allowed to use that model"),
        (404, "model 'm' was not found"),
        (429, "rate limited or out of quota"),
        (400, "the request was rejected: bad shape"),
        (500, "HTTP 500"),
    ],
)
async def test_vendor_errors_become_readable_messages_that_never_contain_the_key(
    make, vendor, status, expected
):
    vendor.reply({"error": {"message": "bad shape", "type": "x", "code": None}}, status=status)
    with pytest.raises(UpstreamError, match=expected) as exc:
        await make().complete(model="m", system="s", messages=HI, tools=[])
    assert "sk-test" not in str(exc.value) and str(exc.value).startswith("OpenAI-compatible API:")


# --- the factory ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "class_name"),
    [
        ("anthropic", "ClaudeProvider"),
        ("google", "GeminiProvider"),
        ("openai", "OpenAICompatProvider"),
    ],
)
async def test_the_factory_builds_the_right_adapter(provider, class_name):
    endpoint = Endpoint(name="e", provider=provider, apiKey="k", models=["m"])
    built = create_provider(endpoint)
    assert type(built).__name__ == class_name
    await built.aclose()


def test_a_missing_sdk_gives_an_install_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "sitedesk.chat.providers.claude", raising=False)
    monkeypatch.setitem(
        sys.modules, "anthropic", None
    )  # makes `import anthropic` raise ImportError
    endpoint = Endpoint(name="claude", provider="anthropic", apiKey="k", models=["m"])
    with pytest.raises(
        ChatConfigError, match=r"needs the 'anthropic' package: pip install 'sitedesk\[chat\]'"
    ):
        create_provider(endpoint)
