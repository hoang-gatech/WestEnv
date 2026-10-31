"""Calls the real upstream services. Opt in with:  pytest -m live -s

Loose assertions on purpose: these check that the tools still work against today's APIs and
print how long each took, not that the data has a particular value.
"""

import time

import pytest
from mcp import Client

from sitedesk.mcp_server import mcp
from sitedesk.tools import geocode, hotspots, soil, terrain, water, weather

pytestmark = pytest.mark.live

BIEN_HOA_AIRBASE = (10.978, 106.815)


async def timed(label, coro):
    start = time.perf_counter()
    try:
        return await coro
    finally:
        print(f"\n[live] {label}: {time.perf_counter() - start:.1f} s")


async def test_weather():
    result = await timed("weather", weather.get_weather(*BIEN_HOA_AIRBASE, days=3))
    assert len(result.daily) == 3 and result.current.temperature_c is not None


async def test_terrain():
    result = await timed("terrain", terrain.get_terrain(*BIEN_HOA_AIRBASE))
    assert 0 <= result.elevation_m < 200  # the Dong Nai lowlands


async def test_soil():
    result = await timed("soil (properties + WRB, concurrent)", soil.get_soil(*BIEN_HOA_AIRBASE))
    assert result.depths and result.most_probable_wrb_class
    top = result.depths[0]
    assert 99 <= top.clay_pct + top.silt_pct + top.sand_pct <= 101
    print(f"[live] soil: {result.most_probable_wrb_class}, {top.usda_texture_class}")


async def test_water():
    result = await timed("water (Overpass)", water.get_nearby_water(10.9480, 106.8330, 1000))
    assert result.total_found > 0 and result.nearest is not None
    print(f"[live] water: nearest {result.nearest.kind} at {result.nearest.distance_m} m")


async def test_reverse_geocode():
    result = await timed("reverse geocode (Nominatim)", geocode.reverse_geocode(*BIEN_HOA_AIRBASE))
    assert result.country == "Vietnam"
    print(f"[live] place: {result.province} / {result.locality}")


async def test_hotspots():
    result = await timed("hotspots (local)", hotspots.get_known_hotspots(*BIEN_HOA_AIRBASE))
    assert result.nearest.id == "bien-hoa"


async def test_site_profile_through_mcp():
    async with Client(mcp) as client:
        start = time.perf_counter()
        result = await client.call_tool(
            "get_site_profile", {"lat": BIEN_HOA_AIRBASE[0], "lon": BIEN_HOA_AIRBASE[1]}
        )
        print(f"\n[live] get_site_profile via MCP: {time.perf_counter() - start:.1f} s")
    assert not result.is_error
    for name, section in result.structured_content["sections"].items():
        print(f"[live]   {name:9s} ok={section['ok']} {section['error'] or ''}")
    assert result.structured_content["sections"]["hotspots"]["ok"]
