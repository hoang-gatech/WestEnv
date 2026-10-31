"""Elevation, slope and downslope direction from the Open-Meteo elevation API (Copernicus DEM)."""

import math

from .. import cache, geo, http
from ..errors import UpstreamError
from ..models import Location, TerrainResult, source

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
SOURCE_NAME = "Open-Meteo elevation API (Copernicus GLO-90 DEM, ~90 m grid)"
TTL_S = 30 * 24 * 3600
STEP_M = 100.0  # spacing of the four neighbours sampled around the point
FLAT_BELOW_PERCENT = 1.5  # elevations are whole metres on a 90 m grid: below this is noise


def slope_and_downslope(
    north: float, south: float, east: float, west: float, step_m: float
) -> tuple[float, float | None]:
    """Slope (%) and downslope bearing from four neighbouring elevations (central differences)."""
    dzdx = (east - west) / (2 * step_m)
    dzdy = (north - south) / (2 * step_m)
    slope_percent = 100 * math.hypot(dzdx, dzdy)
    if slope_percent == 0:
        return 0.0, None
    return slope_percent, math.degrees(math.atan2(-dzdx, -dzdy)) % 360


async def get_terrain(lat: float, lon: float) -> TerrainResult:
    lat, lon = geo.snap(*geo.validate(lat, lon), decimals=4)
    n = geo.offset(lat, lon, STEP_M, 0)
    s = geo.offset(lat, lon, -STEP_M, 0)
    e = geo.offset(lat, lon, 0, STEP_M)
    w = geo.offset(lat, lon, 0, -STEP_M)
    pts = [(lat, lon), n, s, e, w]
    params = {
        "latitude": ",".join(f"{p[0]:.6f}" for p in pts),
        "longitude": ",".join(f"{p[1]:.6f}" for p in pts),
    }
    fetched_at, raw = await cache.default.get_or_fetch(
        "terrain",
        f"{lat:.4f},{lon:.4f}",
        TTL_S,
        lambda: http.get_json("Open-Meteo elevation", ELEVATION_URL, params=params),
    )
    elevations = raw.get("elevation") if isinstance(raw, dict) else None
    if not elevations or len(elevations) != 5 or any(v is None for v in elevations):
        raise UpstreamError("Open-Meteo elevation", f"expected 5 elevations, got {raw!r}")
    centre, north, south, east, west = elevations
    slope, bearing = slope_and_downslope(north, south, east, west, STEP_M)
    flat = slope < FLAT_BELOW_PERCENT
    caveats = [
        "Slope and direction are estimated from a ~90 m DEM sampled 100 m around the point: "
        "indicative only, and blind to small features such as bunds, ditches and cuttings."
    ]
    if flat:
        caveats.append(
            f"Slope is below {FLAT_BELOW_PERCENT}%, which is within the DEM's noise: "
            "treat the ground as flat and do not infer a flow direction."
        )
    return TerrainResult(
        location=Location(lat=lat, lon=lon),
        sources=[source(SOURCE_NAME, ELEVATION_URL, fetched_at)],
        caveats=caveats,
        elevation_m=centre,
        slope_percent=round(slope, 1),
        downslope_bearing_deg=None if flat or bearing is None else round(bearing),
        downslope_direction=None if flat or bearing is None else geo.compass(bearing),
        sample_spacing_m=STEP_M,
    )
