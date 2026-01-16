from __future__ import annotations

def test_robust_does_not_waste_hepa_in_mild_with_budget_2():
    """
    In mild conditions, robust should NOT burn the full HEPA budget by default.
    We allow some HEPA (e.g., 0-2h/day), but it should be << heuristic's typical behavior
    if heuristic is running fan/hepa unnecessarily.
    """
    from shield.demo.scenarios import make_demo_forcing, list_demo_scenarios
    from shield.core.state import BuildingProfile
    from shield.evaluation.runner import evaluate_policies
    from datetime import datetime

    scenario = "mild"
    assert scenario in list_demo_scenarios()

    start = datetime(2026, 1, 16, 0, 0, 0)
    forcing = make_demo_forcing(start=start, horizon_hours=72, scenario=scenario)
    building = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)

    rows, _ = evaluate_policies(
        forcing,
        building,
        policies=["robust", "heuristic"],
        robust_kwargs={"seed": 42},
        include_baselines=False,
        hepa_budget_h_per_day=2.0,
    )

    r = {x["policy"]: x for x in rows}
    assert "robust" in r and "heuristic" in r

    # Guardrail: robust should not run HEPA the whole time in mild with a small budget.
    # (Budget is 2h/day -> 6h total over 3 days, so max is 6 anyway. We want it to be <= 6 (budget ok),
    # but also preferably <= 3 total hours (half the budget) as a "don't waste" default.)
    assert int(r["robust"]["hepa_hours"]) <= 6
    assert int(r["robust"]["hepa_hours"]) <= 3


def test_robust_avoids_heat_regression_in_heat_only_budget_2():
    """
    In heat-only scenario, robust shouldn't make heat outcomes worse than heuristic
    when constrained to a small HEPA budget (HEPA shouldn't even matter much here).
    """
    from shield.demo.scenarios import make_demo_forcing, list_demo_scenarios
    from shield.core.state import BuildingProfile
    from shield.evaluation.runner import evaluate_policies
    from datetime import datetime

    scenario = "heat_only"
    assert scenario in list_demo_scenarios()

    start = datetime(2026, 1, 16, 0, 0, 0)
    forcing = make_demo_forcing(start=start, horizon_hours=72, scenario=scenario)
    building = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)

    rows, _ = evaluate_policies(
        forcing,
        building,
        policies=["robust", "heuristic"],
        robust_kwargs={"seed": 42},
        include_baselines=False,
        hepa_budget_h_per_day=2.0,
    )

    r = {x["policy"]: x for x in rows}
    robust_heat = float(r["robust"]["heat_hours"])
    heur_heat = float(r["heuristic"]["heat_hours"])

    # Guardrail: robust should be no worse (<=) on heat hours
    assert robust_heat <= heur_heat


def test_robust_beats_or_matches_pm_in_smoke_only_budget_2():
    """
    In smoke-only, robust should at least match heuristic on PM.
    (Today it ties at 0% improvement; after v2 it should beat it modestly.)
    """
    from shield.demo.scenarios import make_demo_forcing, list_demo_scenarios
    from shield.core.state import BuildingProfile
    from shield.evaluation.runner import evaluate_policies
    from datetime import datetime

    scenario = "smoke_only"
    assert scenario in list_demo_scenarios()

    start = datetime(2026, 1, 16, 0, 0, 0)
    forcing = make_demo_forcing(start=start, horizon_hours=72, scenario=scenario)
    building = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)

    rows, _ = evaluate_policies(
        forcing,
        building,
        policies=["robust", "heuristic"],
        robust_kwargs={"seed": 42},
        include_baselines=False,
        hepa_budget_h_per_day=2.0,
    )

    r = {x["policy"]: x for x in rows}
    robust_pm = float(r["robust"]["pm_minutes"])
    heur_pm = float(r["heuristic"]["pm_minutes"])

    assert robust_pm <= heur_pm
