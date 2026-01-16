from datetime import datetime

from shield.core.state import BuildingProfile
from shield.demo.scenarios import make_demo_forcing
from shield.optimizer.robust_greedy import robust_greedy_plan


def test_robust_plan_length_and_types():
    b = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=12, scenario="smoke_heat_day")

    plan = robust_greedy_plan(forcing, b, n_ensemble=30, seed=123)
    assert len(plan) == 12
    assert all(hasattr(a, "windows_open") and hasattr(a, "hepa_on") and hasattr(a, "fan_on") for a in plan)


def test_robust_plan_respects_no_hepa():
    b = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=False, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=8, scenario="smoke_heat_day")

    plan = robust_greedy_plan(forcing, b, n_ensemble=20, seed=123)
    assert all(a.hepa_on is False for a in plan)


def test_robust_plan_deterministic_same_seed():
    b = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=10, scenario="smoke_heat_day")

    p1 = robust_greedy_plan(forcing, b, n_ensemble=20, seed=999)
    p2 = robust_greedy_plan(forcing, b, n_ensemble=20, seed=999)

    assert [(a.windows_open, a.hepa_on, a.fan_on) for a in p1] == [(a.windows_open, a.hepa_on, a.fan_on) for a in p2]
