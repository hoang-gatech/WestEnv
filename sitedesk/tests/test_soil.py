import copy

import pytest
import respx

from sitedesk.errors import UpstreamError
from sitedesk.tools import soil

# One point well inside each region of the USDA texture triangle: (sand, silt, clay, class)
TRIANGLE = [
    (92, 5, 3, "sand"),
    (82, 12, 6, "loamy sand"),
    (65, 25, 10, "sandy loam"),
    (40, 40, 20, "loam"),
    (20, 65, 15, "silt loam"),
    (7, 87, 6, "silt"),
    (60, 15, 25, "sandy clay loam"),
    (33, 34, 33, "clay loam"),
    (10, 58, 32, "silty clay loam"),
    (50, 7, 43, "sandy clay"),
    (6, 47, 47, "silty clay"),
    (20, 20, 60, "clay"),
]


@pytest.mark.parametrize(("sand", "silt", "clay", "expected"), TRIANGLE)
def test_texture_class_for_each_region_of_the_triangle(sand, silt, clay, expected):
    assert soil.usda_texture_class(sand, silt, clay) == expected


def test_texture_class_covers_the_whole_triangle():
    """Every whole-percent (sand, silt, clay) combination must land in some class: no gaps."""
    gaps = [
        (sand, 100 - sand - clay, clay)
        for sand in range(101)
        for clay in range(101 - sand)
        if soil.usda_texture_class(sand, 100 - sand - clay, clay) is None
    ]
    assert gaps == []


def test_texture_class_rescales_and_handles_empty_input():
    assert soil.usda_texture_class(66, 68, 66) == "clay loam"  # same proportions as 33/34/33
    assert soil.usda_texture_class(0, 0, 0) is None


def test_parse_properties_scales_units_and_derives_texture(fx):
    depths = soil.parse_properties(fx("soilgrids_properties.json"))
    assert [d.depth for d in depths] == ["0-5cm", "5-15cm", "15-30cm"]
    top = depths[0]
    assert (top.clay_pct, top.silt_pct, top.sand_pct) == (36.9, 33.3, 29.8)
    assert top.ph_h2o == 5.6
    assert top.organic_carbon_g_per_kg == 23.6
    assert top.usda_texture_class == "clay loam"


def test_parse_properties_returns_nothing_for_open_water(fx):
    assert soil.parse_properties(fx("soilgrids_ocean.json")) == []


def test_parse_properties_refuses_to_mislabel_changed_units(fx):
    raw = copy.deepcopy(fx("soilgrids_properties.json"))
    clay = next(layer for layer in raw["properties"]["layers"] if layer["name"] == "clay")
    clay["unit_measure"]["target_units"] = "g/kg"
    with pytest.raises(UpstreamError, match="clay is now reported in 'g/kg'"):
        soil.parse_properties(raw)


def test_parse_classification(fx):
    name, probs = soil.parse_classification(fx("soilgrids_classification.json"))
    assert name == "Acrisols"
    assert [(p.name, p.probability_pct) for p in probs][:2] == [("Acrisols", 22), ("Cambisols", 12)]


@respx.mock
async def test_get_soil_combines_both_endpoints(fx):
    props = respx.get(soil.PROPERTIES_URL).respond(json=fx("soilgrids_properties.json"))
    respx.get(soil.CLASSIFICATION_URL).respond(json=fx("soilgrids_classification.json"))
    result = await soil.get_soil(10.9801, 106.8199)
    assert (result.location.lat, result.location.lon) == (10.98, 106.82)  # snapped to 3 dp
    assert result.most_probable_wrb_class == "Acrisols"
    assert result.depths[0].usda_texture_class == "clay loam"
    assert len(result.sources) == 2
    # the request asked for exactly the properties and depths the parser expects
    q = props.calls.last.request.url.params.multi_items()
    assert {v for k, v in q if k == "property"} == set(soil.EXPECTED_UNITS)
    assert [v for k, v in q if k == "depth"] == list(soil.DEPTHS)
    # a 22% top class is flagged as low confidence
    assert any("Low confidence" in c for c in result.caveats)


@respx.mock
async def test_get_soil_degrades_when_one_endpoint_fails(fx):
    respx.get(soil.PROPERTIES_URL).respond(json=fx("soilgrids_properties.json"))
    respx.get(soil.CLASSIFICATION_URL).respond(500)
    result = await soil.get_soil(10.98, 106.82)
    assert result.depths and result.most_probable_wrb_class is None
    assert any("WRB soil group unavailable" in c for c in result.caveats)
    assert len(result.sources) == 1


@respx.mock
async def test_get_soil_raises_when_both_endpoints_fail():
    respx.get(soil.PROPERTIES_URL).respond(503)
    respx.get(soil.CLASSIFICATION_URL).respond(503)
    with pytest.raises(UpstreamError, match="SoilGrids"):
        await soil.get_soil(10.98, 106.82)


@respx.mock
async def test_get_soil_explains_a_point_with_no_data(fx):
    respx.get(soil.PROPERTIES_URL).respond(json=fx("soilgrids_ocean.json"))
    respx.get(soil.CLASSIFICATION_URL).respond(404)
    result = await soil.get_soil(12.0, 110.5)
    assert result.depths == []
    assert any("no values here" in c for c in result.caveats)


@respx.mock
async def test_get_soil_does_not_hide_programming_errors(fx):
    respx.get(soil.PROPERTIES_URL).respond(json={"unexpected": "shape"})  # KeyError in the parser
    respx.get(soil.CLASSIFICATION_URL).respond(json=fx("soilgrids_classification.json"))
    with pytest.raises(KeyError):
        await soil.get_soil(10.98, 106.82)
