import pytest
import respx

from sitedesk import geo
from sitedesk.tools import geocode, hotspots

BIEN_HOA_AIRBASE = (10.978, 106.815)
HO_CHI_MINH_CITY = (10.8231, 106.6297)
HANOI = (21.0285, 105.8542)


async def test_a_point_at_a_hotspot_finds_it_first():
    result = await hotspots.get_known_hotspots(*BIEN_HOA_AIRBASE)
    assert result.nearest.id == "bien-hoa"
    assert result.nearest.distance_km == 0.0
    assert [h.id for h in result.within_radius] == ["bien-hoa"]


async def test_distance_and_direction_from_ho_chi_minh_city():
    # Hand check: ~20 km east and ~17 km north, so about 26 km to the north-east.
    result = await hotspots.get_known_hotspots(*HO_CHI_MINH_CITY, radius_km=50)
    (hotspot,) = result.within_radius
    assert hotspot.id == "bien-hoa"
    assert hotspot.distance_km == pytest.approx(26.5, abs=1.0)
    assert hotspot.direction == "NE"


async def test_far_point_has_nothing_in_radius_but_still_reports_the_nearest():
    result = await hotspots.get_known_hotspots(*HANOI, radius_km=100)
    assert result.within_radius == []
    assert result.nearest.id == "da-nang"
    assert 550 < result.nearest.distance_km < 650
    assert any("does not mean the site is clean" in c for c in result.caveats)


async def test_radius_widens_the_result_and_sorts_by_distance():
    result = await hotspots.get_known_hotspots(*HO_CHI_MINH_CITY, radius_km=2000)
    ids = [h.id for h in result.within_radius]
    assert ids[0] == "bien-hoa" and set(ids) == {"bien-hoa", "phu-cat", "da-nang"}
    assert [h.distance_km for h in result.within_radius] == sorted(
        h.distance_km for h in result.within_radius
    )


async def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValueError):
        await hotspots.get_known_hotspots(95, 0)


def test_dataset_is_well_formed():
    rows = hotspots.load_dataset()["hotspots"]
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(geo.in_vietnam_bbox(r["lat"], r["lon"]) for r in rows)
    assert all(r["coordinate_basis"] and r["summary"] for r in rows)


# --- reverse geocoding ---------------------------------------------------------------------------


def test_parse_real_nominatim_response(fx):
    place = geocode.parse_place(fx("nominatim_reverse.json"), 10.98, 106.82, fetched_at=0)
    assert place.country == "Vietnam"
    assert place.province == "Đồng Nai"
    assert place.locality == "Tan Trieu Ward"
    assert "Dong Nai" in place.display_name
    assert any("2025" in c for c in place.caveats)


def test_parse_nominatim_no_result():
    place = geocode.parse_place({"error": "Unable to geocode"}, 12.0, 110.5, fetched_at=0)
    assert place.display_name is None and place.province is None and place.address == {}
    assert any("No address found" in c for c in place.caveats)


@respx.mock
async def test_reverse_geocode_identifies_itself_and_caches(monkeypatch, fx):
    monkeypatch.setattr(geocode, "MIN_INTERVAL_S", 0)
    route = respx.get(geocode.settings.nominatim_url).respond(json=fx("nominatim_reverse.json"))
    result = await geocode.reverse_geocode(10.98001, 106.82001)
    request = route.calls.last.request
    assert request.headers["user-agent"].startswith("sitedesk/")  # Nominatim's policy needs one
    assert request.url.params["zoom"] == "14"
    assert result.province == "Đồng Nai"
    await geocode.reverse_geocode(10.98, 106.82)
    assert route.call_count == 1
