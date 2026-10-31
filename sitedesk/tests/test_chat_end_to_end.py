"""The whole chain per vendor: HTTP, real agent loop and MCP tools, real SDK, fake vendor server.

What this adds over the unit tests: the real tools' real JSON Schemas go through each vendor SDK's
serialisation, and a real tool result comes back in each vendor's own format.
"""

import json

import pytest
from fastapi.testclient import TestClient

from sitedesk import api
from sitedesk.chat.service import ChatService

ALL_TOOLS = {
    "get_weather",
    "get_soil",
    "get_terrain",
    "get_nearby_water",
    "get_known_hotspots",
    "reverse_geocode",
    "get_site_profile",
}
ASK = {"message": "What is near here?", "point": {"lat": 10.978, "lon": 106.815}}
ARGS = {"lat": 10.978, "lon": 106.815}


class Claude:
    provider, model, base = "anthropic", "claude-opus-5-5", ""

    @staticmethod
    def first():
        return {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "x",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 400, "output_tokens": 30},
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_known_hotspots", "input": ARGS}
            ],
        }

    @staticmethod
    def last(text):
        return {
            "id": "m2",
            "type": "message",
            "role": "assistant",
            "model": "x",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 900, "output_tokens": 60},
            "content": [{"type": "text", "text": text}],
        }

    @staticmethod
    def tool_names(body):
        return {t["name"] for t in body["tools"]}

    @staticmethod
    def schema(body, name):
        return next(t for t in body["tools"] if t["name"] == name)["input_schema"]

    @staticmethod
    def returned_result(body):
        return json.loads(body["messages"][-1]["content"][0]["content"])


class Gemini:
    provider, model, base = "google", "gemini-flash-latest", ""

    @staticmethod
    def first():
        part = {
            "functionCall": {"id": "fc1", "name": "get_known_hotspots", "args": ARGS},
            "thoughtSignature": "c2ln",
        }
        return {
            "candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 400, "candidatesTokenCount": 30},
        }

    @staticmethod
    def last(text):
        return {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 60},
        }

    @staticmethod
    def tool_names(body):
        return {d["name"] for d in body["tools"][0]["functionDeclarations"]}

    @staticmethod
    def schema(body, name):
        return next(d for d in body["tools"][0]["functionDeclarations"] if d["name"] == name)[
            "parameters_json_schema"
        ]

    @staticmethod
    def returned_result(body):
        return body["contents"][-1]["parts"][0]["functionResponse"]["response"]["result"]


class OpenAI:
    provider, model, base = "openai", "some-model", "/v1"

    @staticmethod
    def first():
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_known_hotspots", "arguments": json.dumps(ARGS)},
        }
        return {
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": "x",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                }
            ],
            "usage": {"prompt_tokens": 400, "completion_tokens": 30, "total_tokens": 430},
        }

    @staticmethod
    def last(text):
        return {
            "id": "c2",
            "object": "chat.completion",
            "created": 1,
            "model": "x",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {"prompt_tokens": 900, "completion_tokens": 60, "total_tokens": 960},
        }

    @staticmethod
    def tool_names(body):
        return {t["function"]["name"] for t in body["tools"]}

    @staticmethod
    def schema(body, name):
        return next(t for t in body["tools"] if t["function"]["name"] == name)["function"][
            "parameters"
        ]

    @staticmethod
    def returned_result(body):
        return json.loads(body["messages"][-1]["content"])


@pytest.mark.parametrize(
    "vendor_kind", [Claude, Gemini, OpenAI], ids=["claude", "gemini", "openai"]
)
def test_a_question_goes_through_the_real_sdk_the_real_tools_and_back(
    vendor, vendor_kind, tmp_path, monkeypatch
):
    path = tmp_path / "sitedesk.yaml"
    path.write_text(
        f"""
endpoints:
  - name: v
    provider: {vendor_kind.provider}
    apiKey: test-key
    baseURL: {vendor.url}{vendor_kind.base}
    models: [{vendor_kind.model}]
"""
    )
    service = ChatService(path)  # builds the real provider: nothing is patched below the HTTP layer
    monkeypatch.setattr(api, "chat_service", lambda: service)
    vendor.reply(vendor_kind.first())
    vendor.reply(vendor_kind.last("Bien Hoa airbase is on site."))

    with TestClient(api.app) as client:
        response = client.post("/api/chat", json=ASK)
        client.portal.call(service.aclose)

    events = [
        (fields["event"], json.loads(fields["data"]))
        for fields in (
            dict(line.split(": ", 1) for line in b.splitlines())
            for b in response.text.strip().split("\n\n")
        )
    ]
    assert [name for name, _ in events] == ["tool_call", "tool_result", "answer", "done"]
    assert events[0][1]["name"] == "get_known_hotspots" and events[1][1]["ok"] is True
    assert events[2][1]["text"] == "Bien Hoa airbase is on site."
    assert (events[3][1]["input_tokens"], events[3][1]["output_tokens"]) == (
        1300,
        90,
    )  # the vendor's own counts

    first, second = (r["json"] for r in vendor.requests)
    assert vendor_kind.tool_names(first) == ALL_TOOLS  # every real tool reached the vendor
    lat = vendor_kind.schema(first, "get_known_hotspots")["properties"]["lat"]
    assert (lat["minimum"], lat["maximum"]) == (
        -90,
        90,
    )  # with the real schema, constraints included
    assert (
        vendor_kind.returned_result(second)["nearest"]["id"] == "bien-hoa"
    )  # a real tool's real result
