"""Vendor-neutral conversation types: each provider adapter converts to and from its own API."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema of the arguments


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str  # JSON text for a success, the error message for a failure
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self


@dataclass
class Turn:
    """One model response."""

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    # The vendor's own representation of this response, sent back verbatim on the next request:
    # Gemini 3 rejects function calls that lose their thought signatures, and Claude needs its
    # thinking blocks returned unchanged.
    raw: Any = None
    refused: str | None = None  # set when the model declined to answer: says why
    truncated: bool = False  # the output limit cut the answer short


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)  # assistant messages
    results: list[ToolResult] = field(default_factory=list)  # tool messages
    raw: Any = None  # for assistant messages: Turn.raw


class Provider(Protocol):
    """One vendor's API. The agent loop only ever sees this interface."""

    async def complete(
        self, *, model: str, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Turn:
        """One request: the conversation so far in, the model's next turn out."""
        ...

    async def aclose(self) -> None: ...
