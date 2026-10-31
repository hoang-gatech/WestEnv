from pathlib import Path

import pytest

from sitedesk.chat import config
from sitedesk.chat.config import ChatConfigError, load_config

EXAMPLE = Path(config.ROOT / "sitedesk.example.yaml")

MINIMAL = """
version: 1
chat:
  defaultEndpoint: claude
endpoints:
  - name: gemini
    provider: google
    apiKey: "${GEMINI_API_KEY}"
    models: [gemini-flash-latest, gemini-pro-latest]
  - name: claude
    provider: anthropic
    apiKey: "${ANTHROPIC_API_KEY}"
    models: [claude-opus-5-5]
"""


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "sitedesk.yaml"
    path.write_text(MINIMAL)
    return path


def test_dotenv_parsing():
    text = """
# a comment
GEMINI_API_KEY=abc123
export ANTHROPIC_API_KEY="quoted value"
SINGLE='single quoted'
WITH_COMMENT=value # trailing comment
EMPTY=
not a pair
"""
    assert config.parse_dotenv(text) == {
        "GEMINI_API_KEY": "abc123",
        "ANTHROPIC_API_KEY": "quoted value",
        "SINGLE": "single quoted",
        "WITH_COMMENT": "value",
        "EMPTY": "",
    }


def test_interpolation_reaches_nested_values_and_blanks_unset_names():
    tree = {"a": "${X}", "b": ["pre-${X}-post", {"c": "${MISSING}"}], "n": 7}
    assert config.interpolate(tree, {"X": "1"}) == {
        "a": "1",
        "b": ["pre-1-post", {"c": ""}],
        "n": 7,
    }


def test_keys_come_from_the_environment(cfg_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    cfg = load_config(cfg_path)
    assert cfg.endpoint("gemini").api_key == "gem-key" and cfg.endpoint("gemini").has_key
    assert not cfg.endpoint("claude").has_key  # unset: the endpoint loads but is not usable


def test_keys_come_from_a_dotenv_file_and_the_environment_wins(cfg_path, monkeypatch):
    (cfg_path.parent / ".env").write_text(
        "GEMINI_API_KEY=from-dotenv\nANTHROPIC_API_KEY=also-dotenv\n"
    )
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")
    cfg = load_config(cfg_path)
    assert cfg.endpoint("gemini").api_key == "from-environment"
    assert cfg.endpoint("claude").api_key == "also-dotenv"


def test_a_literal_key_in_the_yaml_works(cfg_path):
    cfg_path.write_text(MINIMAL.replace('"${GEMINI_API_KEY}"', "AIza-literal"))
    assert load_config(cfg_path).endpoint("gemini").api_key == "AIza-literal"


def test_default_endpoint_selection(cfg_path):
    cfg = load_config(cfg_path)
    assert cfg.endpoint(None).name == "claude"  # chat.defaultEndpoint
    assert cfg.endpoint("gemini").models[0] == "gemini-flash-latest"
    cfg_path.write_text(MINIMAL.replace("  defaultEndpoint: claude\n", ""))
    assert load_config(cfg_path).endpoint(None).name == "gemini"  # else the first one


def test_an_endpoint_never_prints_its_key(cfg_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    endpoint = load_config(cfg_path).endpoint("gemini")
    assert "super-secret-key" not in repr(endpoint) and "super-secret-key" not in str(endpoint)


def test_a_typo_in_a_key_name_is_reported_without_echoing_the_secret(cfg_path):
    cfg_path.write_text(
        MINIMAL.replace("apiKey:", "apiKye:").replace('"${GEMINI_API_KEY}"', "sk-do-not-print")
    )
    with pytest.raises(ChatConfigError) as exc:
        load_config(cfg_path)
    assert "apiKye" in str(exc.value)
    assert "sk-do-not-print" not in str(exc.value)


def test_unknown_provider_lists_the_valid_ones(cfg_path):
    cfg_path.write_text(MINIMAL.replace("provider: google", "provider: bard"))
    with pytest.raises(ChatConfigError, match="endpoints.0.provider.*google.*anthropic.*openai"):
        load_config(cfg_path)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s.replace("name: claude", "name: gemini"), "unique"),
        (
            lambda s: s.replace("defaultEndpoint: claude", "defaultEndpoint: nope"),
            "defaultEndpoint 'nope'",
        ),
        (lambda s: s.replace("models: [claude-opus-5-5]", "models: []"), "models"),
        (lambda s: s.replace("version: 1", "version: 1\nsurprise: true"), "surprise"),
    ],
)
def test_invalid_configs_are_explained(cfg_path, edit, message):
    cfg_path.write_text(edit(MINIMAL))
    with pytest.raises(ChatConfigError, match=message):
        load_config(cfg_path)


def test_invalid_yaml_gives_a_line_number_and_no_content(cfg_path):
    cfg_path.write_text("endpoints:\n  - name: gemini\n    apiKey: [unclosed secret-value\n")
    with pytest.raises(ChatConfigError, match=r"not valid YAML \(line \d+\)") as exc:
        load_config(cfg_path)
    assert "secret-value" not in str(exc.value)


def test_a_missing_file_says_how_to_create_it(tmp_path):
    with pytest.raises(ChatConfigError, match="Copy sitedesk.example.yaml to sitedesk.yaml"):
        load_config(tmp_path / "sitedesk.yaml")


def test_an_empty_file_is_a_config_with_no_endpoints(tmp_path):
    path = tmp_path / "sitedesk.yaml"
    path.write_text("")
    assert load_config(path).endpoints == []


def test_the_shipped_example_stays_valid(monkeypatch):
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_config(EXAMPLE)
    assert [e.name for e in cfg.endpoints] == ["gemini", "claude"]
    assert [e.provider for e in cfg.endpoints] == ["google", "anthropic"]
    assert not any(e.has_key for e in cfg.endpoints)  # it ships with no key in it
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert load_config(EXAMPLE).endpoint("gemini").has_key
