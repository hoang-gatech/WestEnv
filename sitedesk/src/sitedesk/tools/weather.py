"""Current weather and daily forecast from Open-Meteo (gridded model output, no API key)."""

import json
from collections.abc import Sequence
from importlib import resources
from typing import Any

from .. import cache, geo, http
from ..models import (
    CurrentWeather,
    DailyForecast,
    Location,
    WeatherResult,
    source,
)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE_NAME = "Open-Meteo (open-meteo.com), blend of national weather-service models"
TTL_S = 10 * 60
MAX_DAYS = 16

_CURRENT = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
_DAILY = (
    "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "precipitation_probability_max,wind_speed_10m_max"
)

# WMO weather interpretation codes, as used by Open-Meteo
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}  # fmt: skip


def condition(code: int | None) -> str:
    return "unknown" if code is None else _WMO.get(code, f"weather code {code}")


def parse_weather(raw: dict[str, Any], lat: float, lon: float, fetched_at: float) -> WeatherResult:
    cur, daily = raw["current"], raw["daily"]
    days = [
        DailyForecast(
            date=date,
            condition=condition(daily["weather_code"][i]),
            temp_min_c=daily["temperature_2m_min"][i],
            temp_max_c=daily["temperature_2m_max"][i],
            precipitation_mm=daily["precipitation_sum"][i],
            precipitation_probability_pct=daily["precipitation_probability_max"][i],
            wind_max_kmh=daily["wind_speed_10m_max"][i],
        )
        for i, date in enumerate(daily["time"])
    ]
    return WeatherResult(
        location=Location(lat=lat, lon=lon),
        sources=[source(SOURCE_NAME, FORECAST_URL, fetched_at)],
        caveats=[
            "Gridded model output (cell of roughly 2-11 km), not a station measurement; "
            "see model_grid_location for the cell centre actually used.",
            "Forecast skill falls off beyond about 5-7 days; treat later days as trends.",
        ],
        timezone=raw["timezone"],
        model_grid_location=Location(lat=raw["latitude"], lon=raw["longitude"]),
        elevation_m=raw.get("elevation"),
        current=CurrentWeather(
            time_local=cur["time"],
            condition=condition(cur["weather_code"]),
            temperature_c=cur["temperature_2m"],
            humidity_pct=cur["relative_humidity_2m"],
            precipitation_mm=cur["precipitation"],
            wind_speed_kmh=cur["wind_speed_10m"],
        ),
        daily=days,
    )


async def get_weather(lat: float, lon: float, days: int = 7) -> WeatherResult:
    lat, lon = geo.snap(*geo.validate(lat, lon), decimals=2)
    days = max(1, min(days, MAX_DAYS))
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": _CURRENT,
        "daily": _DAILY,
        "timezone": "auto",
        "forecast_days": days,
    }
    fetched_at, raw = await cache.default.get_or_fetch(
        "weather",
        f"{lat:.2f},{lon:.2f},{days}",
        TTL_S,
        lambda: http.get_json("Open-Meteo", FORECAST_URL, params=params),
        persist=False,
    )
    return parse_weather(raw, lat, lon, fetched_at)


# --- map overlay: current conditions in a fixed set of cities ---------------------------------


def load_cities() -> list[dict[str, Any]]:
    text = resources.files("sitedesk").joinpath("data/cities.json").read_text(encoding="utf-8")
    return json.loads(text)["cities"]


def cities_to_geojson(
    cities: Sequence[dict[str, Any]], raw: list[dict[str, Any]]
) -> dict[str, Any]:
    features = []
    for city, wx in zip(cities, raw, strict=True):
        cur = wx["current"]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [city["lon"], city["lat"]]},
                "properties": {
                    "name": city["name"],
                    "temperature_c": cur["temperature_2m"],
                    "precipitation_mm": cur["precipitation"],
                    "wind_speed_kmh": cur["wind_speed_10m"],
                    "condition": condition(cur["weather_code"]),
                    "time_local": cur["time"],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


async def get_weather_overview() -> dict[str, Any]:
    """GeoJSON of current weather for each city in data/cities.json, in one upstream request."""
    cities = load_cities()
    params = {
        "latitude": ",".join(str(c["lat"]) for c in cities),
        "longitude": ",".join(str(c["lon"]) for c in cities),
        "current": _CURRENT,
        "timezone": "auto",
    }
    _, raw = await cache.default.get_or_fetch(
        "weather-overview",
        "all",
        TTL_S,
        lambda: http.get_json("Open-Meteo", FORECAST_URL, params=params),
        persist=False,
    )
    return cities_to_geojson(cities, raw if isinstance(raw, list) else [raw])
