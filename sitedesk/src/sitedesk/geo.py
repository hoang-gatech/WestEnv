"""Small geodesy helpers, enough for point-scale screening without a GIS dependency."""

import math
from collections.abc import Sequence

EARTH_RADIUS_M = 6_371_008.8

# south, west, north, east. Mainland only and deliberately coarse: it also covers parts of
# Laos, Cambodia and the sea, so it answers "plausibly Vietnam", not "inside the border".
VIETNAM_BBOX = (8.3, 102.1, 23.5, 109.6)

_COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")  # fmt: skip

LatLon = tuple[float, float]


def validate(lat: float, lon: float) -> LatLon:
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("lat and lon must be finite numbers")
    if not -90 <= lat <= 90:
        raise ValueError(f"lat {lat} is outside [-90, 90]")
    if not -180 <= lon <= 180:
        raise ValueError(f"lon {lon} is outside [-180, 180]")
    return lat, lon


def snap(lat: float, lon: float, decimals: int) -> LatLon:
    """Round to a grid (2 dp ~ 1.1 km, 3 dp ~ 110 m, 4 dp ~ 11 m) so nearby clicks share a key."""
    return round(lat, decimals), round(lon, decimals)


def in_vietnam_bbox(lat: float, lon: float) -> bool:
    south, west, north, east = VIETNAM_BBOX
    return south <= lat <= north and west <= lon <= east


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360


def compass(bearing: float) -> str:
    return _COMPASS[round(bearing / 22.5) % 16]


def offset(lat: float, lon: float, north_m: float, east_m: float) -> LatLon:
    """Move a short distance (up to a few km) north and east of a point."""
    dlat = math.degrees(north_m / EARTH_RADIUS_M)
    dlon = math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def _to_xy(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """Equirectangular projection around (lat0, lon0), in metres. Fine at the few-km scale."""
    x = math.radians(lon - lon0) * math.cos(math.radians(lat0)) * EARTH_RADIUS_M
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def nearest_on_polyline(
    lat: float, lon: float, line: Sequence[LatLon]
) -> tuple[float, float, float]:
    """Distance (m) from a point to a polyline, and the nearest point on it as (lat, lon)."""
    if not line:
        raise ValueError("polyline has no points")
    pts = [_to_xy(lat, lon, plat, plon) for plat, plon in line]
    if len(pts) == 1:
        pts = pts * 2
    best_d, best_xy = math.inf, pts[0]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:], strict=False):
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, (-x1 * dx - y1 * dy) / seg2))
        px, py = x1 + t * dx, y1 + t * dy
        d = math.hypot(px, py)
        if d < best_d:
            best_d, best_xy = d, (px, py)
    nlat, nlon = offset(lat, lon, best_xy[1], best_xy[0])
    return best_d, nlat, nlon


def point_in_polygon(lat: float, lon: float, ring: Sequence[LatLon]) -> bool:
    """Ray casting on a closed ring of (lat, lon) vertices."""
    inside = False
    n = len(ring)
    for i in range(n):
        (y1, x1), (y2, x2) = ring[i], ring[(i + 1) % n]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside
