import pytest
import respx
from fastapi.testclient import TestClient

from sitedesk.api import app
from sitedesk.tools import terrain


@pytest.fixture
def client():
    with TestClient(app) as c:  # the context manager runs the lifespan on the app's own loop
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_point_endpoint_returns_a_tool_result(client):
    r = client.get("/api/point/hotspots", params={"lat": 10.978, "lon": 106.815})
    assert r.status_code == 200
    body = r.json()
    assert body["nearest"]["id"] == "bien-hoa"
    assert body["location"] == {"lat": 10.978, "lon": 106.815}


def test_unknown_section_is_a_404_that_lists_the_valid_ones(client):
    r = client.get("/api/point/bogus", params={"lat": 10, "lon": 106})
    assert r.status_code == 404
    assert "hotspots" in r.json()["detail"]


@pytest.mark.parametrize(
    "params", [{"lat": 10}, {"lon": 106}, {"lat": 95, "lon": 106}, {"lat": "x", "lon": 106}]
)
def test_bad_coordinates_are_a_422(client, params):
    assert client.get("/api/point/hotspots", params=params).status_code == 422


@respx.mock
def test_upstream_failure_is_a_502_naming_the_service(client):
    respx.get(terrain.ELEVATION_URL).respond(500)
    r = client.get("/api/point/terrain", params={"lat": 10.98, "lon": 106.82})
    assert r.status_code == 502
    assert "Open-Meteo elevation" in r.json()["detail"]


def test_hotspot_layer_is_geojson_with_lon_lat_order(client):
    body = client.get("/api/hotspots").json()
    assert body["type"] == "FeatureCollection" and len(body["features"]) == 3
    bien_hoa = next(f for f in body["features"] if f["properties"]["id"] == "bien-hoa")
    assert bien_hoa["geometry"]["coordinates"] == [106.815, 10.978]
    assert "lat" not in bien_hoa["properties"]
    assert body["about"]


def test_serves_the_map_page_without_shadowing_the_api(client):
    page = client.get("/")
    assert page.status_code == 200 and "SiteDesk" in page.text
    assert client.get("/api/health").status_code == 200


@pytest.mark.parametrize("path", ["/", "/style.css", "/app.js"])
def test_front_end_files_are_revalidated_so_edits_reach_the_browser(client, path):
    # Without Cache-Control browsers cache these by heuristic: a plain reload keeps the old copy.
    first = client.get(path)
    assert first.headers["cache-control"] == "no-cache"
    again = client.get(path, headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304  # still cheap: an unchanged file is not downloaded again
    assert again.headers["cache-control"] == "no-cache"
