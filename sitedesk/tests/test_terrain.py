import httpx
import pytest
import respx

from sitedesk.errors import UpstreamError
from sitedesk.tools import terrain

STEP = 100


@pytest.mark.parametrize(
    ("n", "s", "e", "w", "slope", "downslope"),
    [
        (110, 90, 100, 100, 10.0, 180),  # rises to the north, so water runs south
        (100, 100, 110, 90, 10.0, 270),  # rises to the east, so water runs west
        (90, 110, 100, 100, 10.0, 0),
        (105, 95, 105, 95, 7.07, 225),  # rises to the north-east
    ],
)
def test_slope_and_downslope(n, s, e, w, slope, downslope):
    got_slope, got_dir = terrain.slope_and_downslope(n, s, e, w, STEP)
    assert got_slope == pytest.approx(slope, abs=0.01)
    assert got_dir == pytest.approx(downslope, abs=0.01)


def test_flat_ground_has_no_direction():
    assert terrain.slope_and_downslope(50, 50, 50, 50, STEP) == (0.0, None)


@respx.mock
async def test_get_terrain_reports_slope_and_direction():
    route = respx.get(terrain.ELEVATION_URL).respond(json={"elevation": [50, 60, 40, 50, 50]})
    result = await terrain.get_terrain(10.98, 106.82)
    assert result.elevation_m == 50
    assert result.slope_percent == 10.0
    assert result.downslope_bearing_deg == 180
    assert result.downslope_direction == "S"
    assert len(route.calls.last.request.url.params["latitude"].split(",")) == 5


@respx.mock
async def test_get_terrain_treats_near_zero_slope_as_flat():
    respx.get(terrain.ELEVATION_URL).respond(json={"elevation": [30, 31, 29, 30, 30]})  # 1 %
    result = await terrain.get_terrain(10.0, 105.0)
    assert result.slope_percent == 1.0
    assert result.downslope_bearing_deg is None
    assert result.downslope_direction is None
    assert any("flat" in c for c in result.caveats)


@respx.mock
@pytest.mark.parametrize("body", [{"elevation": [1, 2, 3]}, {"elevation": [1, None, 3, 4, 5]}, {}])
async def test_get_terrain_rejects_unusable_responses(body):
    respx.get(terrain.ELEVATION_URL).respond(json=body)
    with pytest.raises(UpstreamError, match="expected 5 elevations"):
        await terrain.get_terrain(10.98, 106.82)


@respx.mock
async def test_get_terrain_is_cached_by_cell():
    route = respx.get(terrain.ELEVATION_URL).respond(json={"elevation": [50, 60, 40, 50, 50]})
    await terrain.get_terrain(10.98, 106.82)
    await terrain.get_terrain(10.98001, 106.82001)  # same 4 dp cell
    assert route.call_count == 1


@respx.mock
async def test_get_terrain_surfaces_upstream_failure():
    respx.get(terrain.ELEVATION_URL).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(UpstreamError, match="Open-Meteo elevation"):
        await terrain.get_terrain(10.98, 106.82)
