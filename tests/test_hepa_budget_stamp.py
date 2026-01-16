from __future__ import annotations

from datetime import datetime

from shield.core.state import BuildingProfile
from shield.demo.scenarios import make_demo_forcing
from shield.engine.simulate import simulate_demo


def _run(policy: str, hepa_budget: float):
    building = BuildingProfile(
        archetype="classroom",
        floor_area_m2=90.0,
        has_hepa=True,
        has_fan=True,
        occupants=25,
    )
    start = datetime(2026, 1, 16, 0, 0)
    forcing = make_demo_forcing(start=start, horizon_hours=72, scenario="smoke_heat_day")

    policy_kwargs = None
    if policy == "robust_greedy":
        policy_kwargs = {
            "n_ensemble": 20,
            "seed": 123,
            "weights": {
                "lookahead_h": 6,
                "beam_width": 5,
                "rollout_members": 10,
                "risk_lambda": 0.70,
                "cvar_alpha": 0.90,
            },
        }

    return simulate_demo(
        forcing=forcing,
        building=building,
        policy=policy,
        policy_kwargs=policy_kwargs,
        hepa_budget_h_per_day=hepa_budget,
    )


def test_hepa_budget_stamps_action_flags():
    hepa_budget = 2.0
    expected_total_max = int(hepa_budget * 3)  # 72h horizon = 3 days -> max 6 HEPA-hours

    for policy in ["heuristic", "robust_greedy"]:
        run = _run(policy=policy, hepa_budget=hepa_budget)

        hepa_hours = sum(1 for a in run.actions if getattr(a, "hepa_on", False))
        assert hepa_hours <= expected_total_max, (policy, hepa_hours, expected_total_max)

        # If we say we overrode HEPA OFF, the action flag MUST be off too.
        for a in run.actions:
            notes = " ".join(getattr(a, "notes", []) or [])
            if "Override: HEPA OFF (budget exhausted)" in notes:
                assert not getattr(a, "hepa_on", False), (policy, notes)
