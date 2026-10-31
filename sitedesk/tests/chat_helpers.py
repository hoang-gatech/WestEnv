"""Shared fakes for the chat tests."""

from sitedesk.chat.types import ToolCall, Turn

BIEN_HOA = {"lat": 10.978, "lon": 106.815}


class ScriptedProvider:
    """Replays prepared turns (the last one repeats) and records what it was sent."""

    def __init__(self, *turns: Turn, error: Exception | None = None):
        self.turns = list(turns)
        self.error = error
        self.requests: list[dict] = []
        self.closed = False

    async def complete(self, *, model, system, messages, tools):
        self.requests.append(
            {"model": model, "system": system, "messages": list(messages), "tools": list(tools)}
        )
        if self.error:
            raise self.error
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]

    async def aclose(self):
        self.closed = True


def call(id="c1", name="get_known_hotspots", **args):
    return ToolCall(id=id, name=name, args=args or dict(BIEN_HOA))
