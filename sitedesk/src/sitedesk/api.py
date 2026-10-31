"""Web API and static map front end.

    python -m sitedesk.api            # http://127.0.0.1:8000

The map page calls /api/point/{section} once per section, in parallel, so soil (slow) never
holds up weather (fast).
"""

import argparse
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles

from . import http
from .errors import UpstreamError
from .tools import hotspots, weather
from .tools.profile import SECTIONS

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate on every load.

    Starlette sends Last-Modified and ETag but no Cache-Control, so browsers cache by heuristic
    and can keep an old stylesheet or script for hours after an edit: in Chromium a plain reload
    kept serving the old CSS. `no-cache` still allows a 304, so an unchanged file costs a round
    trip and no download.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await http.aclose()


app = FastAPI(title="SiteDesk", lifespan=_lifespan)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/point/{section}")
async def point(
    section: str,
    lat: float = Query(ge=-90, le=90),
    lon: float = Query(ge=-180, le=180),
) -> Any:
    tool = SECTIONS.get(section)
    if tool is None:
        raise HTTPException(404, f"unknown section {section!r}; expected one of {sorted(SECTIONS)}")
    try:
        return await tool(lat, lon)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except UpstreamError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/overview/weather")
async def weather_overview() -> Any:
    try:
        return await weather.get_weather_overview()
    except UpstreamError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/hotspots")
async def hotspot_layer() -> dict[str, Any]:
    dataset = hotspots.load_dataset()
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [h["lon"], h["lat"]]},
                "properties": {k: v for k, v in h.items() if k not in ("lat", "lon")},
            }
            for h in dataset["hotspots"]
        ],
        "about": dataset["about"],
    }


if WEB_DIR.is_dir():  # mounted last so it never shadows /api
    app.mount("/", RevalidatedStaticFiles(directory=WEB_DIR, html=True), name="web")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SiteDesk web API and map")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
