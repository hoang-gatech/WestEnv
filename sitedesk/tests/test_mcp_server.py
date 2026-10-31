"""Drive the real server through a real MCP client session, in memory."""

import respx
from mcp import Client

from sitedesk.mcp_server import mcp
from sitedesk.tools import soil, terrain

EXPECTED_TOOLS = {
    "get_weather",
    "get_soil",
    "get_terrain",
    "get_nearby_water",
    "get_known_hotspots",
    "reverse_geocode",
    "get_site_profile",
}


# The client is opened inside each test, not in a yield-fixture: it owns an anyio task group, which
# must be entered and exited from the same task, and pytest-asyncio runs fixture teardown elsewhere.


async def test_exposes_every_tool_with_a_description_and_schema():
    async with Client(mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == EXPECTED_TOOLS
        for tool in tools.values():
            assert len(tool.description) > 80, (
                f"{tool.name} needs a description the model can choose by"
            )
            assert {"lat", "lon"} <= set(tool.input_schema["properties"])
            assert tool.input_schema["required"][:2] == ["lat", "lon"]
            assert tool.output_schema is not None
            assert tool.annotations.read_only_hint is True


async def test_call_returns_structured_data():
    async with Client(mcp) as client:
        result = await client.call_tool("get_known_hotspots", {"lat": 10.978, "lon": 106.815})
        assert not result.is_error
        assert result.structured_content["nearest"]["id"] == "bien-hoa"
        assert result.structured_content["sources"] and result.structured_content["caveats"]


async def test_out_of_range_arguments_are_rejected_before_the_tool_runs():
    async with Client(mcp) as client:
        result = await client.call_tool("get_known_hotspots", {"lat": 95, "lon": 106.8})
        assert result.is_error


@respx.mock
async def test_upstream_failure_message_reaches_the_model():
    async with Client(mcp) as client:
        respx.get(terrain.ELEVATION_URL).respond(500)
        result = await client.call_tool("get_terrain", {"lat": 10.98, "lon": 106.82})
        assert result.is_error
        # not the SDK's generic "Error executing tool": the model can say what actually failed
        assert "Open-Meteo elevation" in result.content[0].text
        assert "HTTP 500" in result.content[0].text


@respx.mock
async def test_site_profile_survives_a_failing_section(fx):
    async with Client(mcp) as client:
        respx.get(terrain.ELEVATION_URL).respond(json={"elevation": [50, 60, 40, 50, 50]})
        respx.get(soil.PROPERTIES_URL).respond(503)
        respx.get(soil.CLASSIFICATION_URL).respond(503)
        respx.get(url__startswith="https://api.open-meteo.com/v1/forecast").respond(
            json=fx("open_meteo_forecast.json")
        )
        respx.get(url__startswith="https://nominatim.openstreetmap.org").respond(
            json=fx("nominatim_reverse.json")
        )
        result = await client.call_tool("get_site_profile", {"lat": 10.98, "lon": 106.82})
        sections = result.structured_content["sections"]
        assert not result.is_error
        assert set(sections) == {
            "place",
            "weather",
            "terrain",
            "soil",
            "hotspots",
        }  # no water by default
        assert sections["terrain"]["ok"] and sections["terrain"]["data"]["slope_percent"] == 10.0
        assert not sections["soil"]["ok"] and "SoilGrids" in sections["soil"]["error"]
        assert sections["hotspots"]["ok"]


def test_server_instructions_tell_the_model_not_to_overclaim():
    assert "never a survey or laboratory result" in mcp.instructions
