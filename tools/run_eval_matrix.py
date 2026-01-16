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


def _make_robust_kwargs_compat(seed: int) -> Dict[str, Any]:
    """
    Only pass robust kwargs that your installed robust_greedy_plan actually accepts.
    Right now, your run showed it accepts 'seed', so we’ll safely pass that (and only that).
    """
    from shield.optimizer import robust_greedy as rg

    sig = inspect.signature(rg.robust_greedy_plan)
    if "seed" in sig.parameters:
        return {"seed": int(seed)}
    return {}


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=str, default="all", help="comma-separated scenario names, or 'all'")
    ap.add_argument("--archetypes", type=str, default="classroom", help="comma-separated (e.g., classroom,home)")
    ap.add_argument("--areas", type=str, default="90", help="comma-separated areas m2 (e.g., 30,60,90)")
    ap.add_argument("--start-hours", type=str, default="0", help="comma-separated (e.g., 0,6,12,18)")
    ap.add_argument("--horizon-hours", type=int, default=72)
    ap.add_argument("--hepa-budgets", type=str, default="0,2,6", help="comma-separated h/day (0=unlimited)")
    ap.add_argument("--members", type=int, default=25, help="occupants")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    from shield.evaluation.runner import evaluate_policies
    import shield.demo.scenarios as sc

    # scenarios
    if str(args.scenarios).strip().lower() == "all":
        try:
            scenarios = [str(s) for s in sc.list_demo_scenarios()]
        except Exception:
            raise SystemExit("Could not list demo scenarios. Try --scenarios smoke_heat_day,...")
    else:
        scenarios = _parse_csv_list(str(args.scenarios))

    archetypes = _parse_csv_list(str(args.archetypes))
    areas = [float(x) for x in _parse_csv_list(str(args.areas))]
    start_hours = [int(x) for x in _parse_csv_list(str(args.start_hours))]
    budgets_raw = [float(x) for x in _parse_csv_list(str(args.hepa_budgets))]

    policies = ["robust", "heuristic", "always_closed", "always_open", "night_vent", "hepa_always"]

    all_rows: List[Dict[str, Any]] = []
    run_log: List[Dict[str, Any]] = []

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(scenarios) * len(archetypes) * len(areas) * len(start_hours) * len(budgets_raw)
    k = 0

    for scenario in scenarios:
        for archetype in archetypes:
            for area in areas:
                for start_hour in start_hours:
                    forcing = _make_demo_forcing_compat(
                        scenario_name=scenario,
                        start_hour=start_hour,
                        horizon_hours=int(args.horizon_hours),
                    )
                    building = _make_building_compat(
                        archetype=archetype,
                        area_m2=area,
                        has_hepa=True,
                        has_fan=True,
                        occupants=int(args.members),
                    )

                    for b in budgets_raw:
                        k += 1
                        hepa_budget_h_per_day: Optional[float] = float(b)
                        if hepa_budget_h_per_day <= 0.0:
                            hepa_budget_h_per_day = None

                        robust_kwargs = _make_robust_kwargs_compat(int(args.seed))

                        budget_str = "unlimited" if hepa_budget_h_per_day is None else f"{hepa_budget_h_per_day:g}"
                        print(
                            f"[{k:>3}/{total}] scenario={scenario} archetype={archetype} "
                            f"area={area:g} start={start_hour} budget={budget_str}"
                        )

                        try:
                            rows, results = evaluate_policies(
                                forcing,
                                building,
                                policies=policies,
                                robust_kwargs=robust_kwargs if robust_kwargs else None,
                                include_baselines=True,
                                hepa_budget_h_per_day=hepa_budget_h_per_day,
                            )

                            # annotate rows with run metadata
                            for r in rows:
                                rr = dict(r)
                                rr["scenario"] = scenario
                                rr["archetype"] = archetype
                                rr["area_m2"] = float(area)
                                rr["start_hour"] = int(start_hour)
                                rr["horizon_hours"] = int(args.horizon_hours)
                                rr["hepa_budget_h_per_day"] = "" if hepa_budget_h_per_day is None else float(hepa_budget_h_per_day)
                                rr["seed"] = int(args.seed)
                                all_rows.append(rr)

                            run_log.append(
                                {
                                    "scenario": scenario,
                                    "archetype": archetype,
                                    "area_m2": float(area),
                                    "start_hour": int(start_hour),
                                    "horizon_hours": int(args.horizon_hours),
                                    "hepa_budget_h_per_day": None if hepa_budget_h_per_day is None else float(hepa_budget_h_per_day),
                                    "seed": int(args.seed),
                                    "ok": True,
                                    "n_rows": len(rows),
                                    "robust_kwargs": robust_kwargs,
                                }
                            )
                        except Exception as e:
                            run_log.append(
                                {
                                    "scenario": scenario,
                                    "archetype": archetype,
                                    "area_m2": float(area),
                                    "start_hour": int(start_hour),
                                    "horizon_hours": int(args.horizon_hours),
                                    "hepa_budget_h_per_day": None if hepa_budget_h_per_day is None else float(hepa_budget_h_per_day),
                                    "seed": int(args.seed),
                                    "ok": False,
                                    "error": repr(e),
                                }
                            )
                            print("   [ERROR]", repr(e))

    stamp = _now_stamp()
    csv_path = outdir / f"eval_matrix_{stamp}.csv"
    json_path = outdir / f"eval_matrix_{stamp}.json"
    _write_csv(csv_path, all_rows)
    _write_json(
        json_path,
        {
            "rows": all_rows,
            "run_log": run_log,
            "policies": policies,
            "notes": "hepa_budget_h_per_day is blank in CSV when unlimited",
        },
    )

    print(f"\nDONE. Wrote:\n- {csv_path}\n- {json_path}")
    print(f"Rows: {len(all_rows)} | Runs: {len(run_log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())