import pytest

from sitedesk.errors import UpstreamError
from sitedesk.tools import hotspots, profile

BIEN_HOA_AIRBASE = (10.978, 106.815)


@pytest.fixture
def mixed_outcomes(monkeypatch):
    """Two sections succeed, three fail in different ways, and water is the sixth section."""

    async def upstream_down(lat, lon):
        raise UpstreamError("SoilGrids", "HTTP 503 (gave up after 3 attempts)")

    async def bad_input(lat, lon):
        raise ValueError("lat is nonsense")

    async def bug(lat, lon):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(
        profile,
        "SECTIONS",
        {
            "hotspots": hotspots.get_known_hotspots,
            "soil": upstream_down,
            "weather": bad_input,
            "place": bug,
            "water": hotspots.get_known_hotspots,
        },
    )


async def test_failures_stay_in_their_own_section(mixed_outcomes):
    result = await profile.get_site_profile(*BIEN_HOA_AIRBASE)
    s = result.sections
    assert s["hotspots"].ok and s["hotspots"].data["nearest"]["id"] == "bien-hoa"
    assert not s["soil"].ok and "SoilGrids" in s["soil"].error
    assert not s["weather"].ok and s["weather"].error == "lat is nonsense"
    # an unexpected exception is logged but its text is not shown to the model
    assert not s["place"].ok and "secret" not in s["place"].error


async def test_water_is_opt_in_because_it_is_slow(mixed_outcomes):
    assert "water" not in (await profile.get_site_profile(*BIEN_HOA_AIRBASE)).sections
    with_water = await profile.get_site_profile(*BIEN_HOA_AIRBASE, include_water=True)
    assert "water" in with_water.sections


async def test_flags_points_outside_vietnam(mixed_outcomes):
    assert (await profile.get_site_profile(*BIEN_HOA_AIRBASE)).inside_vietnam_bounding_box
    assert not (await profile.get_site_profile(-31.95, 115.86)).inside_vietnam_bounding_box  # Perth


async def test_invalid_coordinates_fail_the_whole_call(mixed_outcomes):
    with pytest.raises(ValueError):
        await profile.get_site_profile(100, 0)


def test_every_section_the_web_ui_uses_is_registered():
    assert set(profile.SECTIONS) == {"place", "weather", "terrain", "soil", "water", "hotspots"}
