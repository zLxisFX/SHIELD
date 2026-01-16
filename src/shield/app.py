"""
SHIELD app entry point (CLI).

Runs:
- Offline Demo Mode (synthetic forcing, no internet)
- Physics-based indoor PM + heat digital twin
- Action planning policy:
    - heuristic (baseline)
    - robust_greedy (UQ-aware risk-based planner)

UPGRADE:
- Robust optimizer supports beam-search lookahead knobs:
    --opt-beam
    --opt-rollout-members
- Evaluation table can include action-use metrics (HEPA/Fan/Windows/Effort)

NOTE (compatibility):
- Some environments may have an older evaluate_policies() that does NOT accept
  hepa_budget_h_per_day. We automatically detect supported kwargs and only pass
  what exists, preventing TypeError crashes.
"""

from __future__ import annotations

import argparse
import inspect
from datetime import datetime
from pathlib import Path

from shield.core.state import BuildingProfile
from shield.demo.scenarios import list_demo_scenarios, make_demo_forcing
from shield.demo.export import export_runresult
from shield.engine.simulate import simulate_demo

# Evaluation imports
from shield.evaluation.runner import evaluate_policies, policy_table_string, export_eval_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHIELD - Offline Demo Mode CLI")

    p.add_argument("--mode", choices=["demo", "live"], default="demo")
    p.add_argument("--scenario", default="smoke_heat_day", help=f"demo scenario ({', '.join(list_demo_scenarios())})")
    p.add_argument("--horizon", type=int, default=72)

    # Start control for determinism
    p.add_argument("--start-hour", type=int, default=None, help="force demo start hour (0-23) for repeatability")

    p.add_argument("--archetype", choices=["apartment", "house", "classroom", "clinic"], default="classroom")
    p.add_argument("--area", type=float, default=90.0)

    p.add_argument("--no-hepa", action="store_true")
    p.add_argument("--no-fan", action="store_true")
    p.add_argument("--occupants", type=int, default=25)

    # Budget: omit/negative => unlimited; 0 => disable HEPA; positive => hours/day
    p.add_argument(
        "--hepa-budget",
        type=float,
        default=None,
        help="HEPA runtime budget in hours per day (omit/negative = unlimited; 0 = disable HEPA)",
    )


    p.add_argument("--out", default="outputs")

    # --- Policy selection ---
    p.add_argument(
        "--policy",
        choices=["heuristic", "robust"],
        default="heuristic",
        help="action planning policy (robust uses UQ-aware optimizer)",
    )

    # --- Robust optimizer knobs ---
    p.add_argument("--opt-ens", type=int, default=120, help="optimizer ensemble size")
    p.add_argument("--opt-seed", type=int, default=123, help="optimizer seed")
    p.add_argument("--opt-lookahead", type=int, default=8, help="optimizer lookahead hours")
    p.add_argument("--opt-beam", type=int, default=10, help="beam width for lookahead planning")
    p.add_argument("--opt-rollout-members", type=int, default=25, help="ensemble members used during lookahead search")
    p.add_argument("--opt-risk", type=float, default=0.55, help="risk lambda: 0=mean only, 1=CVaR only")
    p.add_argument("--opt-cvar", type=float, default=0.90, help="CVaR alpha (e.g., 0.90 => worst 10%)")

    # --- Evaluation mode ---
    p.add_argument("--eval", action="store_true", help="run evaluation table across policies/baselines")
    p.add_argument("--eval-ref", default="heuristic", help="reference policy for % improvements")
    p.add_argument("--eval-export", action="store_true", help="export evaluation JSON to outputs/")
    p.add_argument("--no-eval-baselines", action="store_true", help="disable baseline comparisons in eval mode")

    return p.parse_args()


def _filter_kwargs_for(func, kwargs: dict) -> dict:
    """
    Return only the kwargs that are accepted by `func`.
    Prevents TypeError when runner signatures differ across versions.
    """
    try:
        sig = inspect.signature(func)
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        # If signature introspection fails for any reason, be conservative:
        # pass nothing extra.
        return {}


def main() -> None:
    args = parse_args()

    building = BuildingProfile(
        archetype=args.archetype,
        floor_area_m2=float(args.area),
        has_hepa=(not args.no_hepa),
        has_fan=(not args.no_fan),
        occupants=int(args.occupants),
    )

    if args.mode == "live":
        print("Live mode is not implemented yet. Use --mode demo for now.")
        return

    scenario = str(args.scenario)
    if scenario not in set(list_demo_scenarios()):
        print(f"Unknown scenario: {scenario}")
        print(f"Available: {', '.join(list_demo_scenarios())}")
        return

    # Deterministic-ish start time
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    if args.start_hour is not None:
        h = int(args.start_hour)
        if not (0 <= h <= 23):
            print("--start-hour must be in 0..23")
            return
        start = start.replace(hour=h)

    forcing = make_demo_forcing(start=start, horizon_hours=int(args.horizon), scenario=scenario)

    # Normalize HEPA budget:
    #   None (flag omitted) or negative => unlimited
    #   0 => no HEPA allowed
    #   >0 => hours/day
    hepa_budget_h_per_day = args.hepa_budget
    if hepa_budget_h_per_day is not None:
        hepa_budget_h_per_day = float(hepa_budget_h_per_day)
        if hepa_budget_h_per_day < 0.0:
            hepa_budget_h_per_day = None


    # Robust optimizer kwargs (shared between demo + eval)
    robust_kwargs = {
        "n_ensemble": int(args.opt_ens),
        "seed": int(args.opt_seed),
        "weights": {
            "lookahead_h": int(args.opt_lookahead),
            "beam_width": int(args.opt_beam),
            "rollout_members": int(args.opt_rollout_members),
            "risk_lambda": float(args.opt_risk),
            "cvar_alpha": float(args.opt_cvar),
        },
    }

    # ------------------------
    # Evaluation mode
    # ------------------------
    if bool(args.eval):
        policies = None
        if bool(args.no_eval_baselines):
            policies = ["robust", "heuristic"]

        eval_kwargs = {
            "policies": policies,
            "eval_ref": str(args.eval_ref),
            "robust_kwargs": robust_kwargs,
            "hepa_budget_h_per_day": hepa_budget_h_per_day,
        }

        filtered = _filter_kwargs_for(evaluate_policies, eval_kwargs)

        # If user asked for a budget but runner doesn't support it, warn clearly.
        if hepa_budget_h_per_day is not None and "hepa_budget_h_per_day" not in filtered:
            print(
                "[note] Your installed evaluate_policies() does not accept --hepa-budget yet.\n"
                "       Running eval WITHOUT the HEPA budget constraint.\n"
                "       (Demo runs may still respect it if simulate_demo() supports it.)\n"
            )

        rows, results = evaluate_policies(
            forcing,
            building,
            **filtered,
        )
        print(policy_table_string(rows, eval_ref=str(args.eval_ref)))

        if bool(args.eval_export):
            out_path = export_eval_json(rows, results, out_dir=str(args.out))
            print(f"\nExported eval JSON: {out_path}")
        return

    # ------------------------
    # Single-policy run
    # ------------------------
    policy = "heuristic"
    policy_kwargs = None

    if args.policy == "robust":
        policy = "robust_greedy"
        policy_kwargs = dict(robust_kwargs)

    sim_kwargs = {
        "forcing": forcing,
        "building": building,
        "policy": policy,
        "policy_kwargs": policy_kwargs,
        "hepa_budget_h_per_day": hepa_budget_h_per_day,
    }
    filtered_sim_kwargs = _filter_kwargs_for(simulate_demo, sim_kwargs)

    if hepa_budget_h_per_day is not None and "hepa_budget_h_per_day" not in filtered_sim_kwargs:
        print(
            "[note] Your installed simulate_demo() does not accept --hepa-budget yet.\n"
            "       Running demo WITHOUT the HEPA budget constraint.\n"
        )

    run = simulate_demo(**filtered_sim_kwargs)

    out_dir = Path(args.out)
    txt_path, json_path = export_runresult(run, out_dir=out_dir, prefix=f"shield_plan_{args.policy}")

    print("\nSHIELD demo run complete.")
    print(f"- Policy: {args.policy}")
    print(f"- Demo start: {start.isoformat(timespec='minutes')}")
    print(f"- Exported plan (txt):  {txt_path}")
    print(f"- Exported plan (json): {json_path}")
    print(f"- Indoor PM minutes > WHO {run.pm25_threshold:.0f} µg/m³: {run.minutes_pm25_above_threshold} min")
    print(f"- Heat hours ≥ {run.heat_threshold_c:.0f}°C: {run.hours_heat_above_threshold} h")

    if hepa_budget_h_per_day is not None:
        print(f"- HEPA budget requested: {hepa_budget_h_per_day:.1f} h/day")

    print("\nPreview (next 6 hours):")
    n = min(6, len(run.outputs), len(run.actions))
    for i in range(n):
        o = run.outputs[i]
        a = run.actions[i]
        notes = ", ".join(a.notes) if a.notes else ""
        print(f"{o.t.isoformat(timespec='minutes')} | PM_in={o.pm25_in:5.1f} | T_in={o.temp_in_c:4.1f} | {notes}")


if __name__ == "__main__":
    main()
