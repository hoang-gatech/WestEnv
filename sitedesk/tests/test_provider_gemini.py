"""The Gemini adapter on the real google-genai SDK, against a fake generateContent."""

import pytest
from google import genai
from google.genai import types

from sitedesk.chat.config import Endpoint
from sitedesk.chat.providers.gemini import GeminiProvider
from sitedesk.chat.types import Message, ToolCall, ToolResult, ToolSpec
from sitedesk.errors import UpstreamError

MODEL = "gemini-flash-latest"
SOIL_SCHEMA = {
    "type": "object",
    "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}},
    "required": ["lat", "lon"],
}
TOOLS = [ToolSpec("get_soil", "Soil at a point.", SOIL_SCHEMA)]
HI = [Message(role="user", text="hi")]
SIGNATURE = "c2lnbmF0dXJlLTE="  # base64 of "signature-1": opaque to us, must come back unchanged


def generated(parts, finish="STOP", usage=(100, 10, 0)):
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": finish, "index": 0}
        ],
        "usageMetadata": {
            "promptTokenCount": usage[0],
            "candidatesTokenCount": usage[1],
            "thoughtsTokenCount": usage[2],
            "totalTokenCount": sum(usage),
        },
        "modelVersion": MODEL,
    }


def function_call(name="get_soil", id="fc-1", signature=None, **args):
    part = {"functionCall": {"name": name, "args": args or {"lat": 10.978, "lon": 106.815}}}
    if id:
        part["functionCall"]["id"] = id
    if signature:
        part["thoughtSignature"] = signature
    return part


@pytest.fixture
async def make(vendor):
    made: list[GeminiProvider] = []

    def build(**endpoint_kwargs):
        endpoint = Endpoint(
            name="gemini", provider="google", apiKey="AIza-test", models=[MODEL], **endpoint_kwargs
        )
        client = genai.Client(
            api_key="AIza-test", http_options=types.HttpOptions(base_url=vendor.url)
        )
        made.append(GeminiProvider(endpoint, client=client))
        return made[-1]

    yield build
    for provider in made:
        await provider.aclose()


async def test_request_on_the_wire(make, vendor):
    vendor.reply(generated([{"text": "Hello."}]))
    turn = await make().complete(model=MODEL, system="Be brief.", messages=HI, tools=TOOLS)
    sent = vendor.last
    assert sent["path"] == f"/v1beta/models/{MODEL}:generateContent"
    assert (
        sent["headers"]["x-goog-api-key"] == "AIza-test"
    )  # in a header, so it never lands in a logged URL
    assert "key=" not in sent["path"]
    body = sent["json"]
    assert body["systemInstruction"]["parts"] == [{"text": "Be brief."}]
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    (declaration,) = body["tools"][0]["functionDeclarations"]
    assert declaration["name"] == "get_soil" and declaration["description"] == "Soil at a point."
    # the SDK sends this one field snake_case; the API's JSON parser accepts both spellings
    assert declaration["parameters_json_schema"] == SOIL_SCHEMA
    assert "maxOutputTokens" not in body.get(
        "generationConfig", {}
    )  # none configured: the model's default
    assert (turn.text, turn.calls, turn.refused, turn.truncated) == ("Hello.", [], None, False)
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (100, 10)


async def test_the_function_call_round_trip_keeps_ids_and_thought_signatures(make, vendor):
    provider = make()
    vendor.reply(generated([function_call(signature=SIGNATURE)]))
    turn = await provider.complete(model=MODEL, system="s", messages=HI, tools=TOOLS)
    assert turn.calls == [ToolCall("fc-1", "get_soil", {"lat": 10.978, "lon": 106.815})]

    vendor.reply(generated([{"text": "Acrisols."}]))
    history = [
        *HI,
        Message(role="assistant", calls=turn.calls, raw=turn.raw),
        Message(role="tool", results=[ToolResult("fc-1", "get_soil", '{"wrb":"Acrisols"}')]),
    ]
    await provider.complete(model=MODEL, system="s", messages=history, tools=TOOLS)
    _, model_turn, result_turn = vendor.last["json"]["contents"]
    assert model_turn["role"] == "model"
    assert (
        model_turn["parts"][0]["thoughtSignature"] == SIGNATURE
    )  # Gemini 3 rejects calls that lose it
    assert model_turn["parts"][0]["functionCall"]["id"] == "fc-1"
    assert result_turn == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "id": "fc-1",
                    "name": "get_soil",
                    "response": {"result": {"wrb": "Acrisols"}},
                }
            }
        ],
    }


async def test_parallel_calls_come_back_in_one_turn_in_order(make, vendor):
    provider = make()
    vendor.reply(generated([function_call(id="a", signature=SIGNATURE), function_call(id="b")]))
    turn = await provider.complete(model=MODEL, system="s", messages=HI, tools=TOOLS)
    assert [c.id for c in turn.calls] == ["a", "b"]

    vendor.reply(generated([{"text": "ok"}]))
    history = [
        *HI,
        Message(role="assistant", calls=turn.calls, raw=turn.raw),
        Message(
            role="tool",
            results=[ToolResult("a", "get_soil", "one"), ToolResult("b", "get_soil", "two")],
        ),
    ]
    await provider.complete(model=MODEL, system="s", messages=history, tools=TOOLS)
    parts = vendor.last["json"]["contents"][-1]["parts"]
    assert [p["functionResponse"]["id"] for p in parts] == ["a", "b"]
    assert [p["functionResponse"]["response"] for p in parts] == [
        {"result": "one"},
        {"result": "two"},
    ]


async def test_a_call_without_an_id_gets_a_private_one_that_is_not_sent_back(make, vendor):
    provider = make()
    vendor.reply(generated([function_call(id=None)]))
    turn = await provider.complete(model=MODEL, system="s", messages=HI, tools=TOOLS)
    assert turn.calls[0].id.startswith("g-")  # our events need an id; the API did not give one
    vendor.reply(generated([{"text": "ok"}]))
    history = [
        *HI,
        Message(role="assistant", raw=turn.raw),
        Message(role="tool", results=[ToolResult(turn.calls[0].id, "get_soil", "x")]),
    ]
    await provider.complete(model=MODEL, system="s", messages=history, tools=TOOLS)
    response = vendor.last["json"]["contents"][-1]["parts"][0]["functionResponse"]
    assert "id" not in response and response["name"] == "get_soil"


async def test_a_failed_tool_is_reported_as_an_error_object(make, vendor):
    vendor.reply(generated([{"text": "ok"}]))
    history = [
        *HI,
        Message(
            role="assistant",
            raw=types.Content(
                role="model",
                parts=[
                    types.Part(function_call=types.FunctionCall(id="a", name="get_soil", args={}))
                ],
            ),
        ),
        Message(
            role="tool", results=[ToolResult("a", "get_soil", "SoilGrids: HTTP 503", is_error=True)]
        ),
    ]
    await make().complete(model=MODEL, system="s", messages=history, tools=TOOLS)
    assert vendor.last["json"]["contents"][-1]["parts"][0]["functionResponse"]["response"] == {
        "error": "SoilGrids: HTTP 503"
    }


async def test_plain_text_history_is_sent_with_gemini_roles(make, vendor):
    vendor.reply(generated([{"text": "ok"}]))
    history = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="hello"),
        Message(role="user", text="again"),
    ]
    await make().complete(model=MODEL, system="s", messages=history, tools=[])
    assert [c["role"] for c in vendor.last["json"]["contents"]] == ["user", "model", "user"]
    assert "tools" not in vendor.last["json"]


async def test_thought_summaries_are_not_part_of_the_answer(make, vendor):
    vendor.reply(generated([{"text": "Let me think...", "thought": True}, {"text": "Acrisols."}]))
    turn = await make().complete(model=MODEL, system="s", messages=HI, tools=[])
    assert turn.text == "Acrisols."


async def test_thinking_tokens_count_as_output(make, vendor):
    vendor.reply(generated([{"text": "ok"}], usage=(100, 10, 40)))
    turn = await make().complete(model=MODEL, system="s", messages=HI, tools=[])
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (100, 50)


async def test_a_configured_output_limit_is_sent(make, vendor):
    vendor.reply(generated([{"text": "ok"}]))
    await make(maxOutputTokens=500).complete(model=MODEL, system="s", messages=HI, tools=[])
    assert vendor.last["json"]["generationConfig"]["maxOutputTokens"] == 500


@pytest.mark.parametrize(
    ("finish", "attr", "expected"),
    [
        ("MAX_TOKENS", "truncated", True),
        ("SAFETY", "refused", "blocked: SAFETY"),
        ("PROHIBITED_CONTENT", "refused", "blocked: PROHIBITED_CONTENT"),
    ],
)
async def test_finish_reasons(make, vendor, finish, attr, expected):
    vendor.reply(generated([{"text": "partial"}], finish=finish))
    turn = await make().complete(model=MODEL, system="s", messages=HI, tools=[])
    assert getattr(turn, attr) == expected


async def test_a_malformed_function_call_is_an_error_not_an_empty_answer(make, vendor):
    vendor.reply(generated([], finish="MALFORMED_FUNCTION_CALL"))
    with pytest.raises(UpstreamError, match="malformed tool call"):
        await make().complete(model=MODEL, system="s", messages=HI, tools=TOOLS)


async def test_a_blocked_prompt_is_reported_as_a_refusal(make, vendor):
    vendor.reply(
        {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 5}}
    )
    turn = await make().complete(model=MODEL, system="s", messages=HI, tools=[])
    assert turn.refused == "blocked: SAFETY"


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (400, "API key not valid. Please pass a valid API key.", "API key was rejected"),
        (403, "Permission denied.", "rejected or is not allowed"),
        (404, "models/x is not found", f"model '{MODEL}' was not found"),
        (429, "Resource has been exhausted", "rate limit or quota exceeded"),
        (500, "Internal error", "HTTP 500: Internal error"),
    ],
)
async def test_vendor_errors_become_readable_messages_that_never_contain_the_key(
    make, vendor, status, message, expected
):
    vendor.reply({"error": {"code": status, "message": message, "status": "ERR"}}, status=status)
    with pytest.raises(UpstreamError, match=expected) as exc:
        await make().complete(model=MODEL, system="s", messages=HI, tools=[])
    assert "AIza-test" not in str(exc.value) and str(exc.value).startswith("Gemini:")
