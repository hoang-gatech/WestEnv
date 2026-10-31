"""Soil properties and WRB soil group from ISRIC SoilGrids 2.0 (250 m, machine-learning model)."""

import asyncio
from typing import Any

from .. import cache, geo, http
from ..errors import UpstreamError
from ..models import Location, SoilDepth, SoilResult, WrbClass, source

PROPERTIES_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
CLASSIFICATION_URL = "https://rest.isric.org/soilgrids/v2.0/classification/query"
SOURCE_NAME = "ISRIC SoilGrids 2.0 (250 m, modelled; CC BY 4.0)"
TTL_S = 30 * 24 * 3600
TIMEOUT_S = 45.0  # measured 3-16 s per call on 2026-10-26, so this leaves headroom

DEPTHS = ("0-5cm", "5-15cm", "15-30cm")
# SoilGrids reports each property as an integer scaled by `d_factor`. The units below are the
# ones this module labels its fields with, so a change upstream must fail loudly, not mislabel.
EXPECTED_UNITS = {"clay": "%", "silt": "%", "sand": "%", "phh2o": "-", "soc": "g/kg"}
LOW_CONFIDENCE_BELOW_PCT = 40


def usda_texture_class(sand: float, silt: float, clay: float) -> str | None:
    """USDA soil texture class from sand/silt/clay percentages (rescaled to sum to 100)."""
    total = sand + silt + clay
    if total <= 0:
        return None
    sand, silt, clay = (100 * v / total for v in (sand, silt, clay))
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 2 * clay < 30:
        return "loamy sand"
    if (7 <= clay < 20 and sand > 52) or (clay < 7 and silt < 50):
        return "sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty clay loam"
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 40 and silt >= 40:
        return "silty clay"
    if clay >= 40 and sand <= 45 and silt < 40:
        return "clay"
    return None


def parse_properties(raw: dict[str, Any]) -> list[SoilDepth]:
    """SoilGrids property response -> one SoilDepth per requested depth, in physical units.

    Returns an empty list when every value is null (open water or outside coverage).
    """
    by_depth: dict[str, dict[str, float | None]] = {}
    for layer in raw["properties"]["layers"]:
        name = layer["name"]
        if name not in EXPECTED_UNITS:
            continue
        units = layer["unit_measure"]
        if units["target_units"] != EXPECTED_UNITS[name]:
            raise UpstreamError(
                "SoilGrids",
                f"{name} is now reported in {units['target_units']!r}, "
                f"expected {EXPECTED_UNITS[name]!r}",
            )
        for depth in layer["depths"]:
            value = depth["values"].get("mean")
            scaled = None if value is None else round(value / units["d_factor"], 2)
            by_depth.setdefault(depth["label"], {})[name] = scaled

    depths = []
    for label in DEPTHS:
        v = by_depth.get(label)
        if not v or all(x is None for x in v.values()):
            continue
        texture = None
        if None not in (v.get("sand"), v.get("silt"), v.get("clay")):
            texture = usda_texture_class(v["sand"], v["silt"], v["clay"])
        depths.append(
            SoilDepth(
                depth=label,
                clay_pct=v.get("clay"),
                silt_pct=v.get("silt"),
                sand_pct=v.get("sand"),
                usda_texture_class=texture,
                ph_h2o=v.get("phh2o"),
                organic_carbon_g_per_kg=v.get("soc"),
            )
        )
    return depths


def parse_classification(raw: dict[str, Any]) -> tuple[str | None, list[WrbClass]]:
    probs = [
        WrbClass(name=n, probability_pct=int(p)) for n, p in raw.get("wrb_class_probability") or []
    ]
    return raw.get("wrb_class_name"), probs


async def _fetch_properties(lat: float, lon: float) -> tuple[float, Any]:
    params = {
        "lat": lat,
        "lon": lon,
        "property": list(EXPECTED_UNITS),
        "depth": list(DEPTHS),
        "value": "mean",
    }
    return await cache.default.get_or_fetch(
        "soil-properties",
        f"{lat:.3f},{lon:.3f}",
        TTL_S,
        lambda: http.get_json("SoilGrids", PROPERTIES_URL, params=params, timeout=TIMEOUT_S),
    )


async def _fetch_classification(lat: float, lon: float) -> tuple[float, Any]:
    params = {"lat": lat, "lon": lon, "number_classes": 5}
    return await cache.default.get_or_fetch(
        "soil-wrb",
        f"{lat:.3f},{lon:.3f}",
        TTL_S,
        lambda: http.get_json("SoilGrids", CLASSIFICATION_URL, params=params, timeout=TIMEOUT_S),
    )


async def get_soil(lat: float, lon: float) -> SoilResult:
    lat, lon = geo.snap(*geo.validate(lat, lon), decimals=3)
    props, wrb = await asyncio.gather(
        _fetch_properties(lat, lon), _fetch_classification(lat, lon), return_exceptions=True
    )
    for outcome in (props, wrb):
        if isinstance(outcome, BaseException) and not isinstance(outcome, UpstreamError):
            raise outcome  # a bug, not an upstream failure: do not hide it behind a caveat
    if isinstance(props, UpstreamError) and isinstance(wrb, UpstreamError):
        raise props

    caveats = [
        "SoilGrids is a global model predicting soil at 250 m: values are estimates, not "
        "measurements, and local variation (fill, made ground, contamination) is not represented. "
        "Confirm with site sampling."
    ]
    sources, depths = [], []
    wrb_name, probs = None, []
    if isinstance(props, UpstreamError):
        caveats.append(f"Soil properties unavailable: {props}")
    else:
        fetched_at, raw = props
        depths = parse_properties(raw)
        sources.append(source(SOURCE_NAME + " - properties", PROPERTIES_URL, fetched_at))
        if not depths:
            caveats.append("SoilGrids has no values here (open water, or outside its coverage).")
    if isinstance(wrb, UpstreamError):
        caveats.append(f"WRB soil group unavailable: {wrb}")
    else:
        fetched_at, raw = wrb
        wrb_name, probs = parse_classification(raw)
        sources.append(
            source(SOURCE_NAME + " - WRB classification", CLASSIFICATION_URL, fetched_at)
        )
        if probs and probs[0].probability_pct < LOW_CONFIDENCE_BELOW_PCT:
            caveats.append(
                f"Low confidence in the WRB soil group: the top class has only "
                f"{probs[0].probability_pct}% probability. "
                f"Quote the alternatives, not just the top class."
            )
    return SoilResult(
        location=Location(lat=lat, lon=lon),
        sources=sources,
        caveats=caveats,
        most_probable_wrb_class=wrb_name,
        wrb_class_probabilities=probs,
        depths=depths,
    )
