"""Run several tools for one point and keep going when some of them fail."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from .. import geo
from ..errors import UpstreamError
from ..models import Location, Section, SiteProfile, ToolResult
from .geocode import reverse_geocode
from .hotspots import get_known_hotspots
from .soil import get_soil
from .terrain import get_terrain
from .water import get_nearby_water
from .weather import get_weather

log = logging.getLogger(__name__)

PointTool = Callable[[float, float], Awaitable[ToolResult]]

# The web UI calls these one by one so a slow section (water, soil) never blocks the others.
SECTIONS: dict[str, PointTool] = {
    "place": reverse_geocode,
    "weather": get_weather,
    "terrain": get_terrain,
    "soil": get_soil,
    "water": get_nearby_water,
    "hotspots": get_known_hotspots,
}


def _as_section(outcome: ToolResult | BaseException) -> Section:
    if isinstance(outcome, ToolResult):
        return Section(ok=True, data=outcome.model_dump(mode="json"))
    if isinstance(outcome, UpstreamError | ValueError):
        return Section(ok=False, error=str(outcome))
    log.error("unexpected failure in site profile", exc_info=outcome)
    return Section(ok=False, error="internal error (see server log)")


async def get_site_profile(lat: float, lon: float, include_water: bool = False) -> SiteProfile:
    lat, lon = geo.validate(lat, lon)
    names = [n for n in SECTIONS if include_water or n != "water"]
    outcomes = await asyncio.gather(*(SECTIONS[n](lat, lon) for n in names), return_exceptions=True)
    return SiteProfile(
        location=Location(lat=lat, lon=lon),
        inside_vietnam_bounding_box=geo.in_vietnam_bbox(lat, lon),
        sections={name: _as_section(o) for name, o in zip(names, outcomes, strict=True)},
    )
