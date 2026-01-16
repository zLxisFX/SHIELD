from __future__ import annotations

import argparse
import csv
import inspect
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # Prefer common columns first
    common = [
        "scenario",
        "archetype",
        "area_m2",
        "start_hour",
        "horizon_hours",
        "hepa_budget_h_per_day",
        "seed",
        "policy",
        "pm_over_min",
        "heat_over_h",
        "hepa_hours",
        "fan_hours",
        "window_open_hours",
        "effort_score",
        "pm_pct_vs_ref",
        "heat_pct_vs_ref",
        "effort_pct_vs_ref",
        "ref_policy",
    ]
    keys = sorted({k for r in rows for k in r.keys()})
    header: List[str] = []
    for k in common:
        if k in keys and k not in header:
            header.append(k)
    for k in keys:
        if k not in header:
            header.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        return _safe_str(o)

    path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")


def _pick_scenario_obj(scenarios_mod: Any, scenario_name: str) -> Any:
    if not hasattr(scenarios_mod, "list_demo_scenarios"):
        return None
    try:
        lst = scenarios_mod.list_demo_scenarios()
    except Exception:
        return None

    for s in lst:
        if str(s) == str(scenario_name):
            return s
        nm = getattr(s, "name", None) or getattr(s, "id", None) or getattr(s, "scenario", None)
        if nm and str(nm) == str(scenario_name):
            return s
    return None


def _make_demo_forcing_compat(scenario_name: str, start_hour: int, horizon_hours: int = 72) -> List[Any]:
    """
    Supports current signature:
      make_demo_forcing(start: datetime, horizon_hours: int, scenario: DemoScenario) -> List[ForcingHour]
    """
    import shield.demo.scenarios as sc

    if not hasattr(sc, "make_demo_forcing"):
        raise RuntimeError("shield.demo.scenarios has no make_demo_forcing()")

    fn = sc.make_demo_forcing
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    if params[:3] == ["start", "horizon_hours", "scenario"]:
        scen_obj = _pick_scenario_obj(sc, scenario_name)
        if scen_obj is None:
            scen_obj = str(scenario_name)

        start_dt = datetime(2026, 1, 16, int(start_hour), 0, 0)
        out = fn(start_dt, int(horizon_hours), scen_obj)
        return list(out)

    raise TypeError(f"Unsupported make_demo_forcing signature: {sig}")


def _make_building_compat(
    archetype: str,
    area_m2: float,
    *,
    has_hepa: bool = True,
    has_fan: bool = True,
    occupants: int = 25,
) -> Any:
    import shield.core.state as st

    if not hasattr(st, "BuildingProfile"):
        raise RuntimeError("shield.core.state has no BuildingProfile")

    BP = st.BuildingProfile
    sig = inspect.signature(BP)
    kwargs: Dict[str, Any] = {}

    def put(names: Sequence[str], value: Any) -> None:
        for n in names:
            if n in sig.parameters:
                kwargs[n] = value
                return

    put(["archetype", "building_type", "profile", "kind"], archetype)
    put(["floor_area_m2", "area_m2", "area", "floor_area"], float(area_m2))
    put(["has_hepa", "hepa", "use_hepa"], bool(has_hepa))
    put(["has_fan", "fan", "use_fan"], bool(has_fan))
    put(["occupants", "n_occupants", "people"], int(occupants))

    # Fill any remaining required params if possible
    for p in sig.parameters.values():
        if p.default is not inspect._empty:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in kwargs:
            continue

        name = p.name.lower()
        if "arch" in name:
            kwargs[p.name] = archetype
        elif "area" in name:
            kwargs[p.name] = float(area_m2)
        elif "hepa" in name:
            kwargs[p.name] = bool(has_hepa)
        elif "fan" in name:
            kwargs[p.name] = bool(has_fan)
        elif "occup" in name or "people" in name:
            kwargs[p.name] = int(occupants)
        else:
            raise TypeError(f"Cannot auto-construct BuildingProfile; missing required arg '{p.name}'. Signature: {sig}")

    return BP(**kwargs)


def _pretty_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)\n"

    preferred = [
        "policy",
        "pm_over_min",
        "heat_over_h",
        "hepa_hours",
        "fan_hours",
        "window_open_hours",
        "effort_score",
    ]
    cols = [c for c in preferred if any(c in r for r in rows)]
    extra = sorted({k for r in rows for k in r.keys() if k not in cols})
    cols += extra

    srows = [{c: _safe_str(r.get(c, "")) for c in cols} for r in rows]
    widths = {c: max(len(c), max(len(sr[c]) for sr in srows)) for c in cols}

    lines: List[str] = []
    lines.append(" | ".join(c.ljust(widths[c]) for c in cols))
    lines.append("-+-".join("-" * widths[c] for c in cols))
    for sr in srows:
        lines.append(" | ".join(sr[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines) + "\n"


def _make_robust_kwargs_compat(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build robust_kwargs using the parameter names supported by *your* robust_greedy_plan().
    This prevents errors like: unexpected keyword argument 'lookahead_h'.
    """
    from shield.optimizer import robust_greedy as rg

    plan_fn = rg.robust_greedy_plan
    sig = inspect.signature(plan_fn)
    names = set(sig.parameters.keys())

    def choose(candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if c in names:
                return c
        return None

    out: Dict[str, Any] = {}

    # Canonical values from CLI
    lookahead_val = int(args.opt_lookahead)
    beam_val = int(args.opt_beam)
    rollout_val = int(args.opt_rollout_members)
    risk_val = float(args.opt_risk)
    cvar_val = float(args.opt_cvar)
    seed_val = int(args.seed)

    k = choose(["lookahead_h", "lookahead", "lookahead_hours", "h_lookahead", "lookahead_steps"])
    if k:
        out[k] = lookahead_val

    k = choose(["beam", "beam_width", "beam_size", "beam_k"])
    if k:
        out[k] = beam_val

    k = choose(["rollout_members", "n_rollout_members", "members", "n_members", "ensemble_members"])
    if k:
        out[k] = rollout_val

    k = choose(["risk", "risk_level", "alpha"])
    if k:
        out[k] = risk_val

    k = choose(["cvar", "cvar_alpha", "beta"])
    if k:
        out[k] = cvar_val

    k = choose(["seed", "rng_seed", "random_seed"])
    if k:
        out[k] = seed_val

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--archetype", type=str, required=True)
    ap.add_argument("--area", type=float, required=True)
    ap.add_argument("--start-hour", type=int, default=0)
    ap.add_argument("--horizon-hours", type=int, default=72)

    ap.add_argument("--hepa-budget", type=float, default=0.0, help="HEPA hours/day (0 = unlimited)")
    ap.add_argument("--members", type=int, default=25, help="occupants (and a convenient default for demos)")
    ap.add_argument("--seed", type=int, default=42)

    # robust knobs
    ap.add_argument("--opt-lookahead", type=int, default=12)
    ap.add_argument("--opt-beam", type=int, default=10)
    ap.add_argument("--opt-rollout-members", type=int, default=25)
    ap.add_argument("--opt-risk", type=float, default=0.70)
    ap.add_argument("--opt-cvar", type=float, default=0.90)

    ap.add_argument("--outdir", type=str, default="outputs", help="where to write eval csv/json")
    
    # HTML report options
    ap.add_argument("--no-html-report", action="store_true", help="disable HTML report output")
    ap.add_argument("--report-policy", type=str, default="robust", help="policy to feature in the HTML report")
    ap.add_argument("--open-report", action="store_true", help="open the HTML report after writing it")

    args = ap.parse_args()

    from shield.evaluation.runner import evaluate_policies

    scenario = str(args.scenario)
    archetype = str(args.archetype)
    area = float(args.area)
    start_hour = int(args.start_hour)
    horizon_hours = int(args.horizon_hours)

    hepa_budget_h_per_day: Optional[float] = float(args.hepa_budget)
    if hepa_budget_h_per_day <= 0.0:
        hepa_budget_h_per_day = None

    print(
        f"\n== Evaluating scenario={scenario} archetype={archetype} area={area:.1f} "
        f"start={start_hour} horizon={horizon_hours} =="
    )

    forcing = _make_demo_forcing_compat(scenario_name=scenario, start_hour=start_hour, horizon_hours=horizon_hours)
    building = _make_building_compat(
        archetype=archetype,
        area_m2=area,
        has_hepa=True,
        has_fan=True,
        occupants=int(args.members),
    )

    robust_kwargs = _make_robust_kwargs_compat(args)

    policies = ["robust", "heuristic", "always_closed", "always_open", "night_vent", "hepa_always"]

    print("   policies:", ", ".join(policies))
    print("   forcing:", f"len={len(forcing)} first={forcing[0] if forcing else None}")
    print("   building:", building)
    if hepa_budget_h_per_day is None:
        print("   HEPA budget: unlimited")
    else:
        print(f"   HEPA budget: {hepa_budget_h_per_day:.2f} h/day")

    # show the robust kwargs we will actually pass (and their names)
    print("   robust_kwargs:", robust_kwargs)

    rows, results = evaluate_policies(
        forcing,
        building,
        policies=policies,
        robust_kwargs=robust_kwargs if robust_kwargs else None,
        include_baselines=True,
        hepa_budget_h_per_day=hepa_budget_h_per_day,
    )

    if not rows:
        print("\n[warn] No rows returned from evaluate_policies().")
        print('        Next step: python -c "import inspect; import shield.evaluation.runner as r; print(inspect.getsource(r.evaluate_policies)[:400])"')
        return 2

    print("\n" + _pretty_table(rows))

    outdir = Path(str(args.outdir))
    stamp = _now_stamp()
    csv_path = outdir / f"eval_{scenario}_{archetype}_A{int(area)}_H{start_hour}_M{int(args.members)}_{stamp}.csv"
    json_path = outdir / f"eval_{scenario}_{archetype}_A{int(area)}_H{start_hour}_M{int(args.members)}_{stamp}.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, {"rows": rows, "results": results, "robust_kwargs": robust_kwargs})

    print(f"Wrote:\n- {csv_path}\n- {json_path}")

    # Optional: write a judge-friendly HTML report
    if not getattr(args, "no_html_report", False):
        try:
            from shield.reporting.html_report import write_html_report
            import webbrowser

            report_path = outdir / (
                f"report_{scenario}_{archetype}_A{int(area)}_H{start_hour}_M{int(args.members)}_{stamp}.html"
            )

            meta = {
                "scenario": scenario,
                "archetype": archetype,
                "area_m2": area,
                "start_hour": start_hour,
                "horizon_hours": int(args.horizon_hours),
                "members": int(args.members),
                "seed": int(args.seed),
                "hepa_budget_h_per_day": ("unlimited" if hepa_budget_h_per_day is None else hepa_budget_h_per_day),
                "robust_kwargs": _safe_str(robust_kwargs),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            }

            write_html_report(
                report_path,
                rows=rows,
                results=results,
                meta=meta,
                focus_policy=getattr(args, "report_policy", "robust"),
                ref_policy="heuristic",
            )

            print(f"- {report_path}")
            if getattr(args, "open_report", False):
                webbrowser.open(report_path.resolve().as_uri())

        except Exception as e:
            print(f"[warn] Failed to write HTML report: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())