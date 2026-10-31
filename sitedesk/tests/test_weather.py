import pytest
import respx

from sitedesk.tools import weather


def test_condition_labels():
    assert weather.condition(0) == "clear sky"
    assert weather.condition(63) == "moderate rain"
    assert weather.condition(95) == "thunderstorm"
    assert weather.condition(123) == "weather code 123"
    assert weather.condition(None) == "unknown"


def test_parse_real_open_meteo_response(fx):
    result = weather.parse_weather(fx("open_meteo_forecast.json"), 10.98, 106.82, fetched_at=0)
    assert result.timezone == "Asia/Ho_Chi_Minh"
    assert len(result.daily) == 3
    day = result.daily[0]
    assert day.temp_min_c <= day.temp_max_c
    assert day.precipitation_mm >= 0
    assert 0 <= (day.precipitation_probability_pct or 0) <= 100
    assert result.current.condition != "unknown"
    # the model grid cell is reported separately from the requested point
    assert result.model_grid_location.lat == pytest.approx(10.98, abs=0.1)
    assert any("not a station measurement" in c for c in result.caveats)


@respx.mock
async def test_get_weather_sends_the_expected_request_and_caches(fx):
    route = respx.get(weather.FORECAST_URL).respond(json=fx("open_meteo_forecast.json"))
    result = await weather.get_weather(10.9812, 106.8188, days=3)
    params = route.calls.last.request.url.params
    assert (params["latitude"], params["longitude"]) == ("10.98", "106.82")  # snapped to 2 dp
    assert params["forecast_days"] == "3"
    assert params["timezone"] == "auto"
    assert len(result.daily) == 3
    await weather.get_weather(10.9799, 106.8201, days=3)
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize(("asked", "sent"), [(99, "16"), (0, "1"), (-5, "1")])
async def test_forecast_days_are_clamped(fx, asked, sent):
    route = respx.get(weather.FORECAST_URL).respond(json=fx("open_meteo_forecast.json"))
    await weather.get_weather(10.98, 106.82, days=asked)
    assert route.calls.last.request.url.params["forecast_days"] == sent


@respx.mock
async def test_overview_is_one_request_for_all_cities(fx):
    cities = weather.load_cities()
    current_only = {"current": fx("open_meteo_forecast.json")["current"]}
    route = respx.get(weather.FORECAST_URL).respond(json=[current_only] * len(cities))
    geojson = await weather.get_weather_overview()
    assert route.call_count == 1
    assert len(geojson["features"]) == len(cities)
    hanoi = next(f for f in geojson["features"] if f["properties"]["name"] == "Hanoi")
    assert hanoi["geometry"]["coordinates"] == [105.8542, 21.0285]  # GeoJSON order: lon, lat
    assert hanoi["properties"]["temperature_c"] == current_only["current"]["temperature_2m"]


def test_cities_are_all_plausibly_in_vietnam():
    from sitedesk import geo

    cities = weather.load_cities()
    assert len(cities) > 30
    assert len({c["name"] for c in cities}) == len(cities)
    assert all(geo.in_vietnam_bbox(c["lat"], c["lon"]) for c in cities)
