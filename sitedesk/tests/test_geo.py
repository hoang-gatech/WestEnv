import math

import pytest

from sitedesk import geo


def test_haversine_one_degree_of_latitude():
    assert geo.haversine_m(0, 0, 1, 0) == pytest.approx(111_195, abs=50)


def test_haversine_hanoi_to_ho_chi_minh_city():
    # Hand check: 10.2 deg of latitude ~ 1135 km, plus ~83 km east, gives ~1138 km.
    km = geo.haversine_m(21.0285, 105.8542, 10.8231, 106.6297) / 1000
    assert 1130 < km < 1145


@pytest.mark.parametrize(
    ("target", "expected"),
    [((1, 0), 0), ((0, 1), 90), ((-1, 0), 180), ((0, -1), 270)],
)
def test_bearing_cardinal_directions(target, expected):
    assert geo.bearing_deg(0, 0, *target) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("bearing", "name"),
    [(0, "N"), (45, "NE"), (90, "E"), (202.5, "SSW"), (270, "W"), (359, "N"), (350, "N")],
)
def test_compass(bearing, name):
    assert geo.compass(bearing) == name


def test_offset_moves_the_requested_distance():
    lat, lon = geo.offset(10.0, 106.0, 100, 0)
    assert geo.haversine_m(10.0, 106.0, lat, lon) == pytest.approx(100, abs=0.5)
    lat, lon = geo.offset(10.0, 106.0, 0, 100)
    assert geo.haversine_m(10.0, 106.0, lat, lon) == pytest.approx(100, abs=0.5)


def test_nearest_on_polyline_perpendicular():
    north_100m = geo.offset(0, 0, 100, 0)[0]
    line = [(north_100m, -0.01), (north_100m, 0.01)]
    dist, nlat, nlon = geo.nearest_on_polyline(0, 0, line)
    assert dist == pytest.approx(100, abs=0.5)
    assert nlat == pytest.approx(north_100m, abs=1e-6)
    assert nlon == pytest.approx(0, abs=1e-6)


def test_nearest_on_polyline_clamps_to_the_end_of_a_segment():
    dist, _, nlon = geo.nearest_on_polyline(0, 0, [(0, 0.001), (0, 0.002)])
    assert dist == pytest.approx(geo.haversine_m(0, 0, 0, 0.001), abs=0.5)
    assert nlon == pytest.approx(0.001, abs=1e-6)


def test_nearest_on_polyline_single_point_and_empty():
    dist, _, _ = geo.nearest_on_polyline(0, 0, [(0, 0.001)])
    assert dist == pytest.approx(111.2, abs=0.5)
    with pytest.raises(ValueError):
        geo.nearest_on_polyline(0, 0, [])


def test_point_in_polygon():
    square = [(-0.001, -0.001), (-0.001, 0.001), (0.001, 0.001), (0.001, -0.001)]
    assert geo.point_in_polygon(0, 0, square)
    assert not geo.point_in_polygon(0, 0.01, square)
    assert not geo.point_in_polygon(0.01, 0, square)


@pytest.mark.parametrize(
    ("lat", "lon"), [(math.nan, 0), (0, math.inf), (91, 0), (-91, 0), (0, 181), (0, -181)]
)
def test_validate_rejects_bad_coordinates(lat, lon):
    with pytest.raises(ValueError):
        geo.validate(lat, lon)


def test_validate_accepts_the_edges():
    assert geo.validate(90, 180) == (90, 180)


def test_snap():
    assert geo.snap(10.98049, 106.81951, 3) == (10.98, 106.82)


def test_vietnam_bounding_box():
    assert geo.in_vietnam_bbox(21.03, 105.85)  # Hanoi
    assert geo.in_vietnam_bbox(9.18, 105.15)  # Ca Mau
    assert not geo.in_vietnam_bbox(-33.87, 151.21)  # Sydney
    assert not geo.in_vietnam_bbox(31.95, 115.86)  # Perth
