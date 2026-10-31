import asyncio
import json

import httpx
import pytest
import respx

from sitedesk import cache, http
from sitedesk.errors import UpstreamError

URL = "https://api.example.test/data"


# --- cache ---------------------------------------------------------------------------------------


async def test_second_lookup_is_served_from_cache():
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        return {"n": calls}

    c = cache.Cache(None)
    first = await c.get_or_fetch("ns", "k", 60, fetch)
    second = await c.get_or_fetch("ns", "k", 60, fetch)
    assert first == second and calls == 1


async def test_expired_entries_are_refetched():
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        return calls

    c = cache.Cache(None)
    await c.get_or_fetch("ns", "k", 0, fetch)
    _, value = await c.get_or_fetch("ns", "k", 0, fetch)
    assert value == 2


async def test_concurrent_lookups_share_one_fetch():
    started = asyncio.Event()
    calls = 0

    async def slow_fetch():
        nonlocal calls
        calls += 1
        started.set()
        await asyncio.sleep(0.05)
        return "value"

    c = cache.Cache(None)
    results = await asyncio.gather(*(c.get_or_fetch("ns", "k", 60, slow_fetch) for _ in range(5)))
    assert calls == 1
    assert {r[1] for r in results} == {"value"}


async def test_failures_reach_every_waiter_and_are_not_cached():
    calls = 0

    async def failing_fetch():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        raise UpstreamError("svc", "down")

    c = cache.Cache(None)
    outcomes = await asyncio.gather(
        *(c.get_or_fetch("ns", "k", 60, failing_fetch) for _ in range(3)), return_exceptions=True
    )
    assert all(isinstance(o, UpstreamError) for o in outcomes) and calls == 1
    with pytest.raises(UpstreamError):
        await c.get_or_fetch("ns", "k", 60, failing_fetch)
    assert calls == 2  # the failure was not remembered


async def test_one_cancelled_caller_does_not_cancel_the_shared_fetch():
    async def fetch():
        await asyncio.sleep(0.05)
        return "done"

    c = cache.Cache(None)
    first = asyncio.create_task(c.get_or_fetch("ns", "k", 60, fetch))
    second = asyncio.create_task(c.get_or_fetch("ns", "k", 60, fetch))
    await asyncio.sleep(0)
    first.cancel()
    assert (await second)[1] == "done"


async def test_disk_persistence_survives_a_new_process(tmp_path):
    async def fetch():
        return {"soil": "Acrisols"}

    await cache.Cache(tmp_path).get_or_fetch("soil", "10.98,106.82", 60, fetch)

    async def must_not_run():
        raise AssertionError("should have been read from disk")

    _, value = await cache.Cache(tmp_path).get_or_fetch("soil", "10.98,106.82", 60, must_not_run)
    assert value == {"soil": "Acrisols"}


async def test_corrupt_disk_entry_is_a_miss_not_a_crash(tmp_path):
    c = cache.Cache(tmp_path)
    await c.get_or_fetch("ns", "k", 60, lambda: asyncio.sleep(0, "v"))
    (path,) = tmp_path.rglob("*.json")
    path.write_text("{not json")
    _, value = await cache.Cache(tmp_path).get_or_fetch(
        "ns", "k", 60, lambda: asyncio.sleep(0, "fresh")
    )
    assert value == "fresh"


async def test_unserialisable_value_still_returns(tmp_path):
    _, value = await cache.Cache(tmp_path).get_or_fetch(
        "ns", "k", 60, lambda: asyncio.sleep(0, {1, 2})
    )
    assert value == {1, 2}
    assert not list(tmp_path.rglob("*.json"))


async def test_persist_false_never_touches_disk(tmp_path):
    await cache.Cache(tmp_path).get_or_fetch(
        "ns", "k", 60, lambda: asyncio.sleep(0, 1), persist=False
    )
    assert list(tmp_path.iterdir()) == []


def test_disk_entries_are_plain_json(tmp_path):
    c = cache.Cache(tmp_path)
    c._write("ns", "ns:k", (123.0, {"a": 1}))
    (path,) = tmp_path.rglob("*.json")
    assert json.loads(path.read_text()) == {"key": "ns:k", "t": 123.0, "v": {"a": 1}}


# --- http ----------------------------------------------------------------------------------------


@respx.mock
async def test_returns_decoded_json():
    respx.get(URL).respond(json={"ok": True})
    assert await http.get_json("svc", URL) == {"ok": True}


@respx.mock
async def test_sends_an_identifying_user_agent():
    route = respx.get(URL).respond(json={})
    await http.get_json("svc", URL)
    assert route.calls.last.request.headers["user-agent"].startswith("sitedesk/")


@respx.mock
async def test_retries_transient_errors_then_succeeds():
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.ReadTimeout(""),
            httpx.Response(200, json={"ok": 1}),
        ]
    )
    assert await http.get_json("svc", URL) == {"ok": 1}
    assert route.call_count == 3


@respx.mock
async def test_gives_up_with_a_message_that_names_the_service():
    route = respx.get(URL).respond(502)
    with pytest.raises(UpstreamError, match=r"svc: HTTP 502 \(gave up after 3 attempts\)"):
        await http.get_json("svc", URL)
    assert route.call_count == 3


@respx.mock
async def test_client_errors_are_not_retried():
    route = respx.get(URL).respond(404, text="no such thing")
    with pytest.raises(UpstreamError, match="HTTP 404.*no such thing"):
        await http.get_json("svc", URL)
    assert route.call_count == 1


@respx.mock
async def test_timeouts_become_upstream_errors_with_a_readable_name():
    respx.get(URL).mock(side_effect=httpx.ConnectTimeout(""))
    with pytest.raises(UpstreamError, match="ConnectTimeout"):
        await http.get_json("svc", URL, retries=0)


@respx.mock
async def test_invalid_json_is_an_upstream_error():
    respx.get(URL).respond(200, text="<html>maintenance</html>")
    with pytest.raises(UpstreamError, match="not valid JSON"):
        await http.get_json("svc", URL)


@respx.mock
async def test_backoff_doubles_and_retry_after_is_honoured_up_to_a_cap(monkeypatch):
    delays = []

    async def record(seconds):
        delays.append(seconds)

    monkeypatch.setattr(http.asyncio, "sleep", record)
    monkeypatch.setattr(http, "BACKOFF_S", 0.5)
    respx.get(URL).mock(side_effect=[httpx.Response(503)] * 3)
    with pytest.raises(UpstreamError):
        await http.get_json("svc", URL)
    assert delays == [0.5, 1.0]

    delays.clear()
    respx.get(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(200, json={}),
        ]
    )
    await http.get_json("svc", URL)
    assert delays == [3.0]

    delays.clear()
    respx.get(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "9999"}),
            httpx.Response(200, json={}),
        ]
    )
    await http.get_json("svc", URL)
    assert delays == [http.MAX_RETRY_AFTER_S]


@respx.mock
async def test_post_sends_form_data():
    route = respx.post(URL).respond(json={"ok": 1})
    assert await http.post_form_json("svc", URL, {"data": "[out:json];"}) == {"ok": 1}
    assert route.calls.last.request.content == b"data=%5Bout%3Ajson%5D%3B"
