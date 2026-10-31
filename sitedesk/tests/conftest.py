import json
from pathlib import Path

import pytest

from sitedesk import cache, http

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fx():
    """Load a recorded upstream response from tests/fixtures."""

    def load(name: str):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    return load


@pytest.fixture(autouse=True)
def isolated(monkeypatch, request):
    """No disk cache between tests, and no real backoff sleeps unless a test goes live."""
    monkeypatch.setattr(cache, "default", cache.Cache(None))
    if "live" not in request.keywords:
        monkeypatch.setattr(http, "BACKOFF_S", 0.0)


@pytest.fixture(autouse=True)
async def close_http_client():
    yield
    await http.aclose()
