from datetime import datetime

from shield.core.state import BuildingProfile
from shield.demo.scenarios import make_demo_forcing
from shield.engine.simulate import simulate_demo


def test_simulate_demo_with_robust_policy_runs():
    b = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=12, scenario="smoke_heat_day")

    run = simulate_demo(forcing, b, policy="robust_greedy", policy_kwargs={"n_ensemble": 20, "seed": 123})
    assert run.horizon_hours == 12
    assert len(run.actions) == 12
    assert len(run.outputs) == 12
