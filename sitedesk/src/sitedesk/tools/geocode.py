"""Place names for a point from Nominatim (OpenStreetMap) reverse geocoding."""

import asyncio
import time
from typing import Any

from .. import cache, geo, http
from ..config import settings
from ..models import Location, PlaceResult, source

SOURCE_NAME = "OpenStreetMap Nominatim (ODbL)"
TTL_S = 30 * 24 * 3600
MIN_INTERVAL_S = 1.0  # Nominatim's usage policy: at most one request per second

_lock = asyncio.Lock()
_last_call = 0.0

_LOCALITY_KEYS = (
    "suburb",
    "quarter",
    "village",
    "town",
    "city_district",
    "neighbourhood",
    "hamlet",
)


async def _throttled_fetch(params: dict[str, Any]) -> Any:
    global _last_call
    async with _lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            return await http.get_json("Nominatim", settings.nominatim_url, params=params)
        finally:
            _last_call = time.monotonic()


def parse_place(raw: dict[str, Any], lat: float, lon: float, fetched_at: float) -> PlaceResult:
    address = {k: v for k, v in (raw.get("address") or {}).items() if isinstance(v, str)}
    caveats = [
        "Administrative names follow OpenStreetMap, which may lag Vietnam's 2025 provincial and "
        "commune mergers; check against current official names before using them in a report."
    ]
    if raw.get("error"):
        caveats.append("No address found here (open water, or outside the mapped area).")
    return PlaceResult(
        location=Location(lat=lat, lon=lon),
        sources=[source(SOURCE_NAME, settings.nominatim_url, fetched_at)],
        caveats=caveats,
        display_name=raw.get("display_name"),
        country=address.get("country"),
        province=address.get("state") or address.get("province") or address.get("city"),
        locality=next((address[k] for k in _LOCALITY_KEYS if k in address), None),
        address=address,
    )


async def reverse_geocode(lat: float, lon: float) -> PlaceResult:
    lat, lon = geo.snap(*geo.validate(lat, lon), decimals=4)
    params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 14, "accept-language": "en"}
    fetched_at, raw = await cache.default.get_or_fetch(
        "place", f"{lat:.4f},{lon:.4f}", TTL_S, lambda: _throttled_fetch(params)
    )
    return parse_place(raw, lat, lon, fetched_at)
