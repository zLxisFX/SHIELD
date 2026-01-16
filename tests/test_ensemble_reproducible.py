from datetime import datetime

from shield.core.state import BuildingProfile
from shield.demo.scenarios import make_demo_forcing
from shield.uq.ensemble import run_ensemble


def test_ensemble_reproducible_same_seed():
    building = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=12, scenario="smoke_heat_day")

    r1 = run_ensemble(forcing, building, n=50, seed=999)
    r2 = run_ensemble(forcing, building, n=50, seed=999)

    assert len(r1.bands) == len(r2.bands) == 12
    # Check a couple representative quantiles match exactly (deterministic)
    assert r1.bands[0].pm50 == r2.bands[0].pm50
    assert r1.bands[5].t90 == r2.bands[5].t90


def test_ensemble_changes_with_different_seed():
    building = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=12, scenario="smoke_heat_day")

    r1 = run_ensemble(forcing, building, n=50, seed=1000)
    r2 = run_ensemble(forcing, building, n=50, seed=2000)

    # Very likely different; check at least one value differs
    assert r1.bands[0].pm50 != r2.bands[0].pm50 or r1.bands[0].t50 != r2.bands[0].t50
