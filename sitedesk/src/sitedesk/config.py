"""Process-wide settings, read from the environment once at import."""

import os
from dataclasses import dataclass
from pathlib import Path

from . import __version__

DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"


def _cache_dir() -> Path | None:
    raw = os.environ.get("SITEDESK_CACHE_DIR")
    if raw is None:
        return Path.home() / ".cache" / "sitedesk"
    return Path(raw) if raw else None  # an empty value turns the disk cache off


@dataclass(frozen=True)
class Settings:
    user_agent: str
    cache_dir: Path | None
    overpass_urls: tuple[str, ...]
    nominatim_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        # OSM's usage policies ask for an identifying User-Agent with a way to reach the operator.
        contact = os.environ.get("SITEDESK_CONTACT", "").strip()
        overpass = os.environ.get("SITEDESK_OVERPASS_URLS", DEFAULT_OVERPASS_URL)
        return cls(
            user_agent=f"sitedesk/{__version__}" + (f" ({contact})" if contact else ""),
            cache_dir=_cache_dir(),
            overpass_urls=tuple(u.strip() for u in overpass.split(",") if u.strip()),
            nominatim_url=os.environ.get("SITEDESK_NOMINATIM_URL", DEFAULT_NOMINATIM_URL),
        )


settings = Settings.from_env()
