from datetime import datetime

from shield.core.state import BuildingProfile
from shield.demo.scenarios import make_demo_forcing
from shield.evaluation.baselines import baseline_actions, list_baselines


def test_baselines_build_schedules():
    b = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=24, scenario="smoke_heat_day")

    for name in list_baselines():
        acts = baseline_actions(forcing, b, name)
        assert len(acts) == 24
        assert all(a.t is not None for a in acts)


def test_baseline_respects_no_hepa():
    b = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=False, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=12, scenario="smoke_heat_day")

    acts = baseline_actions(forcing, b, "hepa_always")
    assert all(a.hepa_on is False for a in acts)
