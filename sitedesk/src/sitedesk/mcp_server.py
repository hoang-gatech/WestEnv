"""MCP server exposing the SiteDesk tools.

    python -m sitedesk.mcp_server                       # stdio, for Claude Code / Claude Desktop
    python -m sitedesk.mcp_server --transport streamable-http --port 8001

The docstrings below are what the model reads when choosing a tool, so they say what each tool
returns, in which units, how slow it is, and what it must not be used for.
"""

import argparse
import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, ParamSpec, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import http, models
from .errors import UpstreamError
from .tools import geocode, hotspots, profile, soil, terrain, water, weather

INSTRUCTIONS = """\
SiteDesk gives point-scale site-screening data for Vietnam: weather, soil, terrain, nearby surface
water, place names and known dioxin hotspots. Pass WGS84 decimal degrees. Every result lists its
sources and caveats: quote the caveats that matter to the question. The data comes from public
datasets and models, so it is screening-level context, never a survey or laboratory result.
Say so when you summarise it, and never invent values for a layer whose tool failed.
"""

Lat = Annotated[
    float, Field(ge=-90, le=90, description="Latitude, decimal degrees WGS84, e.g. 10.978")
]
Lon = Annotated[
    float, Field(ge=-180, le=180, description="Longitude, decimal degrees WGS84, e.g. 106.815")
]

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=True)

P = ParamSpec("P")
R = TypeVar("R")


@asynccontextmanager
async def _lifespan(_server: MCPServer) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await http.aclose()


mcp = MCPServer("sitedesk", instructions=INSTRUCTIONS, lifespan=_lifespan)


def tool(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Register a read-only tool, turning expected failures into errors the model can read.

    MCPServer only passes a message through to the model for ToolError; anything else becomes a
    generic "Error executing tool", which would hide "SoilGrids timed out" from Claude.
    """

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await fn(*args, **kwargs)
        except (UpstreamError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    mcp.tool(annotations=READ_ONLY)(wrapper)
    return wrapper


@tool
async def get_weather(
    lat: Lat,
    lon: Lon,
    days: Annotated[int, Field(ge=1, le=16, description="Forecast days, 1-16")] = 7,
) -> models.WeatherResult:
    """Current weather and a daily forecast for a point (Open-Meteo model output).

    Returns temperature (degC), humidity (%), rainfall (mm), wind (km/h), a condition label and,
    per day, min/max temperature, rainfall total and rain probability. This is a gridded model
    estimate, not a station reading; the cell centre is reported. Use it for work-window planning
    and for rainfall that could mobilise contaminants. Fast (about 1 s).
    """
    return await weather.get_weather(lat, lon, days)


@tool
async def get_soil(lat: Lat, lon: Lon) -> models.SoilResult:
    """Soil at a point from ISRIC SoilGrids (250 m, modelled, not measured).

    Returns the most probable WRB soil group with class probabilities (often low: quote the
    alternatives), and for 0-5, 5-15 and 15-30 cm: clay/silt/sand (%), the USDA texture class,
    pH (H2O) and organic carbon (g/kg). Texture, organic carbon and pH are the properties that
    govern how contaminants sorb, leach and persist. SLOW on the first call for a location
    (about 5-20 s); repeats are cached. Not a substitute for site sampling.
    """
    return await soil.get_soil(lat, lon)


@tool
async def get_terrain(lat: Lat, lon: Lon) -> models.TerrainResult:
    """Elevation (m), slope (%) and downslope direction at a point, from a ~90 m DEM.

    Use it for runoff and drainage context: which way surface water is likely to move and how
    steeply. Slopes under about 1.5% are within the DEM's noise and are reported as flat with no
    direction. Fast (about 1 s). Blind to small features such as bunds, drains and cuttings.
    """
    return await terrain.get_terrain(lat, lon)


@tool
async def get_nearby_water(
    lat: Lat,
    lon: Lon,
    radius_m: Annotated[int, Field(ge=50, le=5000, description="Search radius in metres")] = 1000,
) -> models.WaterResult:
    """Rivers, streams, canals, drains and water bodies near a point, from OpenStreetMap.

    Returns the nearest features with kind, name, distance (m) and compass direction. OpenStreetMap
    is incomplete in rural Vietnam, so an empty result does NOT mean there is no water nearby. This
    is the SLOWEST tool: the first call for a location can take up to a minute (the Overpass API),
    repeats are cached. Prefer get_site_profile (which skips it) unless water proximity matters.
    """
    return await water.get_nearby_water(lat, lon, radius_m)


@tool
async def get_known_hotspots(
    lat: Lat,
    lon: Lon,
    radius_km: Annotated[float, Field(gt=0, le=2000, description="Radius in km")] = 100.0,
) -> models.HotspotsResult:
    """Distance and direction from a point to known dioxin hotspots in Vietnam.

    Covers a short curated list of the best-known sites (Bien Hoa, Da Nang, Phu Cat). Returns those
    within the radius plus the nearest one regardless of distance. A site being far from every
    listed hotspot says nothing about whether it is contaminated. Instant; works offline.
    """
    return await hotspots.get_known_hotspots(lat, lon, radius_km)


@tool
async def reverse_geocode(lat: Lat, lon: Lon) -> models.PlaceResult:
    """Province, locality and address for a point, from OpenStreetMap Nominatim.

    Use it to name a clicked location in a report. Administrative names may lag Vietnam's 2025
    provincial and commune mergers. Rate-limited to one request per second; usually fast.
    """
    return await geocode.reverse_geocode(lat, lon)


@tool
async def get_site_profile(
    lat: Lat,
    lon: Lon,
    include_water: Annotated[
        bool, Field(description="Also query nearby surface water (can add up to a minute)")
    ] = False,
) -> models.SiteProfile:
    """Everything above for one point in a single call: place, weather, terrain, soil, hotspots.

    Sections that fail are reported with an error and do not stop the others. Surface water is
    left out unless include_water is true because of its latency. Use this for "tell me about this
    site"; use the single-purpose tools for a specific question. The first call for a location
    takes as long as the slowest section (soil, 5-20 s).
    """
    return await profile.get_site_profile(lat, lon, include_water)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SiteDesk MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)  # stderr, so it never corrupts the stdio stream
    kwargs: dict[str, Any] = {}
    if args.transport == "streamable-http":
        kwargs = {"host": args.host, "port": args.port}
    mcp.run(args.transport, **kwargs)


if __name__ == "__main__":
    main()
