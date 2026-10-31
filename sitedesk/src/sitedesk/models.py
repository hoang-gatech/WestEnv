"""Result models for every tool.

Field names carry their units (`temperature_c`, `distance_m`) so neither the model nor a reader
has to guess. Every result says where it came from and what it cannot be trusted for.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Location(BaseModel):
    lat: float
    lon: float


class Source(BaseModel):
    name: str
    url: str
    retrieved_at: datetime


def source(name: str, url: str, fetched_at: float) -> Source:
    return Source(name=name, url=url, retrieved_at=datetime.fromtimestamp(fetched_at, tz=UTC))


class ToolResult(BaseModel):
    location: Location = Field(
        description="The point actually queried, after snapping to the cache grid"
    )
    sources: list[Source]
    caveats: list[str] = Field(
        default_factory=list, description="Limits the reader must keep in mind"
    )


# --- weather ---------------------------------------------------------------------------------


class CurrentWeather(BaseModel):
    time_local: str
    condition: str
    temperature_c: float | None
    humidity_pct: float | None
    precipitation_mm: float | None
    wind_speed_kmh: float | None


class DailyForecast(BaseModel):
    date: str
    condition: str
    temp_min_c: float | None
    temp_max_c: float | None
    precipitation_mm: float | None
    precipitation_probability_pct: float | None
    wind_max_kmh: float | None


class WeatherResult(ToolResult):
    timezone: str
    model_grid_location: Location
    elevation_m: float | None
    current: CurrentWeather
    daily: list[DailyForecast]


# --- soil ------------------------------------------------------------------------------------


class WrbClass(BaseModel):
    name: str
    probability_pct: int


class SoilDepth(BaseModel):
    depth: str
    clay_pct: float | None
    silt_pct: float | None
    sand_pct: float | None
    usda_texture_class: str | None
    ph_h2o: float | None
    organic_carbon_g_per_kg: float | None


class SoilResult(ToolResult):
    most_probable_wrb_class: str | None
    wrb_class_probabilities: list[WrbClass]
    depths: list[SoilDepth]


# --- terrain ---------------------------------------------------------------------------------


class TerrainResult(ToolResult):
    elevation_m: float
    slope_percent: float
    downslope_bearing_deg: float | None = Field(
        description="Direction water runs downhill; null when flat"
    )
    downslope_direction: str | None
    sample_spacing_m: float


# --- surface water ---------------------------------------------------------------------------


class WaterFeature(BaseModel):
    kind: str
    name: str | None
    distance_m: int = Field(
        description="To the nearest point on the feature; 0 if the point is inside it"
    )
    bearing_deg: int | None
    direction: str | None
    osm_id: str


class WaterResult(ToolResult):
    radius_m: int
    nearest: WaterFeature | None
    features: list[WaterFeature] = Field(description="Nearest first, one entry per named feature")
    total_found: int


# --- known hotspots --------------------------------------------------------------------------


class Hotspot(BaseModel):
    id: str
    name: str
    province: str
    kind: str
    summary: str
    lat: float
    lon: float
    coordinate_basis: str
    distance_km: float
    direction: str


class HotspotsResult(ToolResult):
    radius_km: float
    within_radius: list[Hotspot]
    nearest: Hotspot


# --- place name ------------------------------------------------------------------------------


class PlaceResult(ToolResult):
    display_name: str | None
    country: str | None
    province: str | None
    locality: str | None
    address: dict[str, str]


# --- aggregate -------------------------------------------------------------------------------


class Section(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class SiteProfile(BaseModel):
    location: Location
    inside_vietnam_bounding_box: bool
    sections: dict[str, Section]
