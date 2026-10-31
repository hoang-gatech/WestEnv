"""LibreChat-style YAML config for the chat assistant: endpoints, models and API keys.

    version: 1
    chat: {defaultEndpoint: gemini, maxToolRounds: 6}
    endpoints:
      - name: gemini
        provider: google               # google | anthropic | openai (any OpenAI-compatible API)
        apiKey: "${GEMINI_API_KEY}"    # an environment variable (recommended) or the key itself
        models: [gemini-flash-latest]  # the first one is the default

As in LibreChat, `${NAME}` is filled from the environment, then from a `.env` file next to the
config. Keys never leave the server, and nothing in this module puts a key into an error message.
"""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parents[3]  # the project root, as in api.WEB_DIR
DEFAULT_PATH = ROOT / "sitedesk.yaml"

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ChatConfigError(ValueError):
    """The chat config is missing or invalid. The message says how to fix it."""


def parse_dotenv(text: str) -> dict[str, str]:
    """KEY=VALUE lines, with optional `export`, quotes and # comments."""
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        env[key] = value
    return env


def interpolate(node: Any, env: dict[str, str]) -> Any:
    """Replace ${NAME} in every string of a parsed YAML tree; an unset name becomes empty."""
    if isinstance(node, str):
        return _VAR.sub(lambda m: env.get(m.group(1), ""), node)
    if isinstance(node, list):
        return [interpolate(v, env) for v in node]
    if isinstance(node, dict):
        return {k: interpolate(v, env) for k, v in node.items()}
    return node


class Endpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(min_length=1, max_length=40)
    provider: Literal["google", "anthropic", "openai"]
    api_key: str = Field(
        "", alias="apiKey", repr=False
    )  # repr=False: never print a key by accident
    base_url: str | None = Field(None, alias="baseURL")
    models: list[str] = Field(min_length=1)
    max_output_tokens: int | None = Field(None, alias="maxOutputTokens", ge=64)
    refusal_fallbacks: bool = Field(True, alias="refusalFallbacks")

    @property
    def has_key(self) -> bool:
        return bool(self.api_key.strip())


class ChatSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    default_endpoint: str | None = Field(None, alias="defaultEndpoint")
    max_tool_rounds: int = Field(6, alias="maxToolRounds", ge=1, le=12)
    system_prompt: str = Field("", alias="systemPrompt")


class ChatConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: int = 1
    chat: ChatSettings = Field(default_factory=ChatSettings)
    endpoints: list[Endpoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_names(self) -> "ChatConfig":
        names = [e.name for e in self.endpoints]
        if len(set(names)) != len(names):
            raise ValueError("endpoint names must be unique")
        default = self.chat.default_endpoint
        if default is not None and default not in names:
            raise ValueError(
                f"chat.defaultEndpoint {default!r} is not one of the endpoints {names}"
            )
        return self

    def endpoint(self, name: str | None) -> Endpoint:
        """The named endpoint, or the default one (chat.defaultEndpoint, else the first)."""
        wanted = (
            name
            or self.chat.default_endpoint
            or (self.endpoints[0].name if self.endpoints else None)
        )
        for endpoint in self.endpoints:
            if endpoint.name == wanted:
                return endpoint
        raise ChatConfigError(
            f"unknown endpoint {wanted!r}; configured: {[e.name for e in self.endpoints]}"
        )


def _describe(exc: ValidationError) -> str:
    # loc + message only: pydantic's `input` would echo the offending value, which may be a key
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'config'}: {e['msg']}" for e in exc.errors()
    )


def load_config(path: Path = DEFAULT_PATH) -> ChatConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ChatConfigError(
            f"No chat config ({path.name}). "
            "Copy sitedesk.example.yaml to sitedesk.yaml and add an API key."
        ) from None
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1})" if mark else ""
        raise ChatConfigError(f"{path.name} is not valid YAML{where}") from None
    if not isinstance(raw, dict):
        raise ChatConfigError(f"{path.name} must be a YAML mapping with 'endpoints' in it")
    for section in ("chat", "endpoints"):
        if (
            section in raw and raw[section] is None
        ):  # a bare `chat:` (everything under it commented out)
            del raw[section]
    dotenv = path.with_name(".env")
    env = {
        **(parse_dotenv(dotenv.read_text(encoding="utf-8")) if dotenv.is_file() else {}),
        **os.environ,
    }
    try:
        return ChatConfig.model_validate(interpolate(raw, env))
    except ValidationError as exc:
        raise ChatConfigError(f"{path.name}: {_describe(exc)}") from None
