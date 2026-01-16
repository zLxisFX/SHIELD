from datetime import datetime

from shield.core.state import BuildingProfile
from shield.demo.scenarios import make_demo_forcing
from shield.evaluation.runner import evaluate_policies


def test_evaluate_policies_returns_rows():
    b = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime(2026, 1, 15, 0, 0), horizon_hours=24, scenario="smoke_heat_day")

    rows, results = evaluate_policies(forcing, b, robust_kwargs={"n_ensemble": 20, "seed": 123})
    assert len(rows) >= 2
    assert "heuristic" in results
    assert "robust" in results
