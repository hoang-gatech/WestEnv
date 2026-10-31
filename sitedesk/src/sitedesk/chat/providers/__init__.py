"""One adapter per vendor, each on that vendor's official SDK, imported only when used."""

from ..config import ChatConfigError, Endpoint
from ..types import Provider

_PACKAGES = {"anthropic": "anthropic", "google": "google-genai", "openai": "openai"}


def create_provider(endpoint: Endpoint) -> Provider:
    try:
        if endpoint.provider == "anthropic":
            from .claude import ClaudeProvider

            return ClaudeProvider(endpoint)
        if endpoint.provider == "google":
            from .gemini import GeminiProvider

            return GeminiProvider(endpoint)
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(endpoint)
    except ImportError:
        raise ChatConfigError(
            f"Endpoint {endpoint.name!r} needs the {_PACKAGES[endpoint.provider]!r} package: "
            "pip install 'sitedesk[chat]'"
        ) from None
