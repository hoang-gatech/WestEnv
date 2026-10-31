import dataclasses

import httpx
import pytest
import respx

from sitedesk import geo
from sitedesk.errors import UpstreamError
from sitedesk.tools import water

POINT = (10.9480, 106.8330)  # near the Dong Nai river at Bien Hoa, where the fixture was recorded
URL = water.settings.overpass_urls[0]


def way(osm_id, tags, points):
    return {
        "type": "way",
        "id": osm_id,
        "tags": tags,
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in points],
    }


def east_west_line_north_of_origin(metres):
    lat = geo.offset(0, 0, metres, 0)[0]
    return [(lat, -0.01), (lat, 0.01)]


def test_query_asks_for_the_right_things():
    q = water.build_query(10.948, 106.833, 750)
    assert "(around:750,10.948,106.833)" in q
    assert q.count("around:750") == 3  # linear waterways, water polygons, water relations
    assert "out geom" in q


def test_real_overpass_response_is_parsed_and_ordered(fx):
    # Regression check on a response recorded 2026-10-26; expected values were read off the
    # parser's own output, so the independent invariants below matter more.
    raw = fx("overpass_water.json")
    features = water.parse_water(raw, *POINT, radius_m=1000)
    assert len(features) == 20
    distances = [f.distance_m for f in features]
    assert distances == sorted(distances)
    nearest = features[0]
    assert (nearest.kind, nearest.distance_m, nearest.direction) == ("stream", 118, "S")
    river = next(f for f in features if f.kind == "river")
    assert river.name == "Sông Cái"


def test_distance_to_a_line_never_exceeds_distance_to_its_nearest_vertex(fx):
    raw = fx("overpass_water.json")
    by_id = {f"{e['type']}/{e['id']}": e for e in raw["elements"]}
    for feature in water.parse_water(raw, *POINT, radius_m=1000):
        vertices = by_id[feature.osm_id]["geometry"]
        nearest_vertex = min(geo.haversine_m(*POINT, v["lat"], v["lon"]) for v in vertices)
        assert feature.distance_m <= nearest_vertex + 1  # +1 m for rounding


def test_smaller_radius_drops_far_features(fx):
    raw = fx("overpass_water.json")
    near = water.parse_water(raw, *POINT, radius_m=500)
    assert len(near) == 11
    assert all(f.distance_m <= 500 for f in near)


def test_direction_and_distance_of_a_known_line():
    raw = {"elements": [way(1, {"waterway": "canal"}, east_west_line_north_of_origin(100))]}
    (feature,) = water.parse_water(raw, 0, 0, radius_m=500)
    assert feature.kind == "canal"
    assert feature.distance_m == pytest.approx(100, abs=1)
    assert (feature.bearing_deg, feature.direction) == (0, "N")


def test_point_inside_a_lake_is_at_distance_zero():
    ring = [(-0.001, -0.001), (-0.001, 0.001), (0.001, 0.001), (0.001, -0.001), (-0.001, -0.001)]
    raw = {"elements": [way(2, {"natural": "water", "water": "lake", "name": "Ho Test"}, ring)]}
    (feature,) = water.parse_water(raw, 0, 0, radius_m=500)
    assert (feature.kind, feature.distance_m) == ("lake", 0)
    assert feature.bearing_deg is None and feature.direction is None


def test_segments_of_one_named_river_are_merged_to_the_nearest():
    near = east_west_line_north_of_origin(80)
    far = east_west_line_north_of_origin(300)
    raw = {
        "elements": [
            way(3, {"waterway": "river", "name": "Song X"}, far),
            way(4, {"waterway": "river", "name": "Song X"}, near),
            way(5, {"waterway": "stream"}, far),  # unnamed: kept separately
        ]
    }
    features = water.parse_water(raw, 0, 0, radius_m=500)
    assert [(f.name, f.distance_m) for f in features] == [("Song X", 80), (None, 300)]
    assert features[0].osm_id == "way/4"


def test_relation_members_contribute_their_geometry():
    member = {"type": "way", "ref": 9, "role": "outer", "geometry": [
        {"lat": p[0], "lon": p[1]} for p in east_west_line_north_of_origin(150)
    ]}  # fmt: skip
    raw = {
        "elements": [
            {"type": "relation", "id": 7, "tags": {"natural": "water"}, "members": [member]}
        ]
    }
    (feature,) = water.parse_water(raw, 0, 0, radius_m=500)
    assert (feature.osm_id, feature.distance_m) == ("relation/7", pytest.approx(150, abs=1))


def test_elements_without_geometry_are_skipped():
    raw = {"elements": [{"type": "way", "id": 1, "tags": {"waterway": "river"}}]}
    assert water.parse_water(raw, 0, 0, radius_m=500) == []


@respx.mock
async def test_get_nearby_water_end_to_end(fx):
    route = respx.post(URL).respond(json=fx("overpass_water.json"))
    result = await water.get_nearby_water(*POINT, radius_m=1000)
    assert result.nearest.distance_m == 118
    assert result.total_found == 20
    assert len(result.features) == water.MAX_FEATURES
    assert any("OpenStreetMap coverage" in c for c in result.caveats)
    assert b"around%3A1000" in route.calls.last.request.content  # form-encoded query
    await water.get_nearby_water(*POINT, radius_m=1000)
    assert route.call_count == 1  # second call served from cache


@respx.mock
async def test_radius_is_clamped_to_protect_the_public_server(fx):
    route = respx.post(URL).respond(json=fx("overpass_water.json"))
    result = await water.get_nearby_water(*POINT, radius_m=1_000_000)
    assert result.radius_m == water.MAX_RADIUS_M
    assert b"around%3A5000" in route.calls.last.request.content


@respx.mock
async def test_empty_result_is_reported_not_invented():
    respx.post(URL).respond(json={"elements": []})
    result = await water.get_nearby_water(*POINT)
    assert result.nearest is None and result.features == [] and result.total_found == 0


@respx.mock
async def test_falls_back_to_the_next_mirror(monkeypatch, fx):
    mirror = "https://mirror.example/api/interpreter"
    patched = dataclasses.replace(water.settings, overpass_urls=(URL, mirror))
    monkeypatch.setattr(water, "settings", patched)
    respx.post(URL).respond(502)
    respx.post(mirror).respond(json=fx("overpass_water.json"))
    result = await water.get_nearby_water(*POINT)
    assert result.total_found == 20


@respx.mock
async def test_reports_failure_when_every_mirror_fails():
    respx.post(URL).mock(side_effect=httpx.ReadTimeout(""))
    with pytest.raises(UpstreamError, match="Overpass: ReadTimeout"):
        await water.get_nearby_water(*POINT)
