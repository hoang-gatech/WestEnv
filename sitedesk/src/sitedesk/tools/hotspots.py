"""Distance to curated, publicly reported dioxin hotspots. Local data, no network."""

import functools
import json
from datetime import UTC, datetime
from importlib import resources
from typing import Any

from .. import geo
from ..models import Hotspot, HotspotsResult, Location, Source

DATASET = "sitedesk/data/hotspots.json (curated seed list)"


@functools.cache
def load_dataset() -> dict[str, Any]:
    text = resources.files("sitedesk").joinpath("data/hotspots.json").read_text(encoding="utf-8")
    return json.loads(text)


def _hotspot(row: dict[str, Any], lat: float, lon: float) -> Hotspot:
    return Hotspot(
        **row,
        distance_km=round(geo.haversine_m(lat, lon, row["lat"], row["lon"]) / 1000, 1),
        direction=geo.compass(geo.bearing_deg(lat, lon, row["lat"], row["lon"])),
    )


async def get_known_hotspots(lat: float, lon: float, radius_km: float = 100.0) -> HotspotsResult:
    lat, lon = geo.snap(*geo.validate(lat, lon), decimals=4)
    dataset = load_dataset()
    ranked = sorted(
        (_hotspot(r, lat, lon) for r in dataset["hotspots"]), key=lambda h: h.distance_km
    )
    return HotspotsResult(
        location=Location(lat=lat, lon=lon),
        sources=[Source(name=DATASET, url="", retrieved_at=datetime.now(UTC))],
        caveats=[
            dataset["about"],
            "No hotspot within the radius does not mean the site is clean: this list only covers "
            "three well-known sites.",
        ],
        radius_km=radius_km,
        within_radius=[h for h in ranked if h.distance_km <= radius_km],
        nearest=ranked[0],
    )
