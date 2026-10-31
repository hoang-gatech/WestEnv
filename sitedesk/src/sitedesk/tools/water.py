"""Nearby rivers, streams, canals and water bodies from OpenStreetMap via the Overpass API."""

from typing import Any

from .. import cache, geo, http
from ..config import settings
from ..errors import UpstreamError
from ..models import Location, WaterFeature, WaterResult, source

SOURCE_NAME = "OpenStreetMap contributors via Overpass API (ODbL)"
TTL_S = 7 * 24 * 3600
# Overpass answered in ~35 s on every call during testing (2026-10-26), and mirrors were worse,
# so the client timeout is generous. Callers that care about latency should not wait on this.
TIMEOUT_S = 60.0
MAX_FEATURES = 8
MAX_RADIUS_M = 5000

_LINEAR = "river|stream|canal|drain|ditch"


def build_query(lat: float, lon: float, radius_m: int) -> str:
    around = f"(around:{radius_m},{lat},{lon})"
    return (
        "[out:json][timeout:25];("
        f'way{around}[waterway~"^({_LINEAR})$"];'
        f"way{around}[natural=water];"
        f"relation{around}[natural=water];"
        ");out geom 200;"
    )


def _kind(tags: dict[str, str]) -> str:
    if tags.get("waterway"):
        return tags["waterway"]
    return tags.get("water") or "water body"  # OSM water=lake|pond|reservoir|river|...


def _lines(element: dict[str, Any]) -> list[list[geo.LatLon]]:
    """Geometry as polylines of (lat, lon). Relations contribute one line per member way."""
    if element["type"] == "way":
        raw_lines = [element.get("geometry") or []]
    else:
        raw_lines = [m.get("geometry") or [] for m in element.get("members", [])]
    return [[(p["lat"], p["lon"]) for p in line] for line in raw_lines if line]


def parse_water(raw: dict[str, Any], lat: float, lon: float, radius_m: int) -> list[WaterFeature]:
    """Overpass elements -> features in radius, nearest first, segments of a named river merged."""
    best: dict[str, WaterFeature] = {}
    for el in raw.get("elements", []):
        lines = _lines(el)
        if not lines:
            continue
        tags = el.get("tags") or {}
        osm_id = f"{el['type']}/{el['id']}"
        nearest = min(
            (geo.nearest_on_polyline(lat, lon, line) for line in lines), key=lambda r: r[0]
        )
        dist, nlat, nlon = nearest
        ring = lines[0]
        closed_water = (
            el["type"] == "way" and tags.get("natural") == "water" and ring[0] == ring[-1]
        )
        inside = closed_water and geo.point_in_polygon(lat, lon, ring)
        if inside:
            dist = 0.0
        if dist > radius_m + 1:
            continue
        bearing = None if dist == 0 else geo.bearing_deg(lat, lon, nlat, nlon)
        feature = WaterFeature(
            kind=_kind(tags),
            name=tags.get("name:en") or tags.get("name"),
            distance_m=round(dist),
            bearing_deg=None if bearing is None else round(bearing),
            direction=None if bearing is None else geo.compass(bearing),
            osm_id=osm_id,
        )
        key = f"{feature.kind}|{feature.name}" if feature.name else osm_id
        if key not in best or feature.distance_m < best[key].distance_m:
            best[key] = feature
    return sorted(best.values(), key=lambda f: f.distance_m)


async def _fetch(query: str) -> Any:
    last: UpstreamError | None = None
    for url in settings.overpass_urls:
        try:
            return await http.post_form_json(
                "Overpass", url, {"data": query}, timeout=TIMEOUT_S, retries=1
            )
        except UpstreamError as exc:
            last = exc  # try the next mirror, if any
    assert last is not None
    raise last


async def get_nearby_water(lat: float, lon: float, radius_m: int = 1000) -> WaterResult:
    lat, lon = geo.snap(*geo.validate(lat, lon), decimals=4)
    radius_m = max(50, min(int(radius_m), MAX_RADIUS_M))
    query = build_query(lat, lon, radius_m)
    fetched_at, raw = await cache.default.get_or_fetch(
        "water", f"{lat:.4f},{lon:.4f},{radius_m}", TTL_S, lambda: _fetch(query)
    )
    features = parse_water(raw, lat, lon, radius_m)
    caveats = [
        "OpenStreetMap coverage is uneven, especially in rural Vietnam: no feature listed does "
        "not mean no water nearby. Ditches, paddies and seasonal channels are often unmapped.",
        "Distance is to the nearest mapped line or shoreline, not to a flow path or groundwater.",
    ]
    if raw.get("remark"):
        caveats.append(f"Overpass reported: {raw['remark']}")
    return WaterResult(
        location=Location(lat=lat, lon=lon),
        sources=[source(SOURCE_NAME, settings.overpass_urls[0], fetched_at)],
        caveats=caveats,
        radius_m=radius_m,
        nearest=features[0] if features else None,
        features=features[:MAX_FEATURES],
        total_found=len(features),
    )
