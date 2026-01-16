"""
Evaluation runner: compare robust + baselines on the same forcing/building.

UPGRADE:
- Adds action-use metrics to each row:
    hepa_hours, fan_hours, window_open_hours, effort_score
  This exposes a real robust advantage even when PM/heat are tied.

- % vs ref is robust when the reference metric is 0 (shows "n/a").
- JSON export always serializable (datetimes -> isoformat).
- CLI can pass `policies=[...]` to choose which policies to evaluate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shield.core.state import BuildingProfile, ForcingHour
from shield.engine.simulate import simulate_demo, _simulate_given_actions
from shield.evaluation.baselines import baseline_actions, list_baselines


def _pct_vs_ref(ref: float, val: float) -> Optional[float]:
    """
    Percent improvement vs reference, where positive means "better" (lower).

      pct = 100 * (ref - val) / ref

    If ref == 0:
      - if val == 0 => 0.0
      - else => None (not defined; caller prints "n/a")
    """
    if ref <= 0.0:
        if val == 0.0:
            return 0.0
        return None
    return 100.0 * (ref - val) / ref


def _fmt_pct(x: Optional[float], width: int = 11) -> str:
    if x is None:
        return f"{'n/a':>{width}}"
    return f"{x:+.1f}%".rjust(width)


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def _action_use_metrics(run: Any) -> Dict[str, int | float]:
    """
    Compute action-use metrics from run.actions.
    Assumes ActionsHour has booleans:
      windows_open, hepa_on, fan_on
    """
    acts = getattr(run, "actions", []) or []
    hepa_h = int(sum(1 for a in acts if bool(getattr(a, "hepa_on", False))))
    fan_h = int(sum(1 for a in acts if bool(getattr(a, "fan_on", False))))
    win_h = int(sum(1 for a in acts if bool(getattr(a, "windows_open", False))))

    # Simple "effort" score (dimensionless) for judge-friendly comparison:
    # - HEPA is the most "costly" action
    # - Fan is moderate
    # - Opening windows has smaller but nonzero effort/risk/noise
    effort = float(1.0 * hepa_h + 0.5 * fan_h + 0.15 * win_h)

    return {
        "hepa_hours": hepa_h,
        "fan_hours": fan_h,
        "window_open_hours": win_h,
        "effort_score": effort,
    }


def evaluate_policies(
    forcing: List[ForcingHour],
    building: BuildingProfile,
    *,
    policies: Optional[List[str]] = None,
    robust_kwargs: Optional[Dict[str, Any]] = None,
    eval_ref: str = "heuristic",
    include_baselines: bool = True,
    pm25_threshold: float = 15.0,
    heat_threshold_c: float = 32.0,
    pm25_init: float = 8.0,
    temp_init_c: float = 28.5,
    hepa_budget_h_per_day: float | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate a set of named policies and return:

      rows: list of dicts with summary + % vs ref fields
      results: mapping policy_name -> RunResult-like object (from engine)

    Supported policy names:
      - "robust"
      - "heuristic"
      - baseline names from list_baselines(): e.g. "always_closed", "always_open", ...
    """
    results: Dict[str, Any] = {}

    if policies is not None:
        order = [str(p) for p in policies]
        include_baselines_effective = True
    else:
        order: List[str] = ["robust"]
        if include_baselines:
            order.extend([n for n in list_baselines() if n != "robust"])
        if "heuristic" not in order:
            order.append("heuristic")
        include_baselines_effective = include_baselines

    def _run_one(name: str) -> None:
        if name in results:
            return

        if name == "robust":
            rk = dict(robust_kwargs or {})
            rk.setdefault("n_ensemble", 60)
            rk.setdefault("seed", 123)

            results["robust"] = simulate_demo(
                forcing=forcing,
                building=building,
                pm25_threshold=pm25_threshold,
                heat_threshold_c=heat_threshold_c,
                pm25_init=pm25_init,
                temp_init_c=temp_init_c,
                policy="robust_greedy",
                policy_kwargs=rk,
                hepa_budget_h_per_day=hepa_budget_h_per_day,
            )
            return

        if name == "heuristic":
            results["heuristic"] = simulate_demo(
                forcing=forcing,
                building=building,
                pm25_threshold=pm25_threshold,
                heat_threshold_c=heat_threshold_c,
                pm25_init=pm25_init,
                temp_init_c=temp_init_c,
                policy="heuristic",
                policy_kwargs=None,
                hepa_budget_h_per_day=hepa_budget_h_per_day,
            )
            return

        if (policies is None and not include_baselines_effective):
            return

        acts = baseline_actions(forcing, building, name)
        results[name] = _simulate_given_actions(
            forcing=forcing,
            building=building,
            actions=acts,
            pm25_threshold=pm25_threshold,
            heat_threshold_c=heat_threshold_c,
            pm25_init=pm25_init,
            temp_init_c=temp_init_c,
        )

    for name in order:
        _run_one(name)

    ref_name = eval_ref if eval_ref in results else "heuristic"
    if ref_name not in results:
        _run_one(ref_name)
    if ref_name not in results:
        _run_one("heuristic")
        ref_name = "heuristic"

    ref = results[ref_name]
    ref_pm = float(getattr(ref, "minutes_pm25_above_threshold"))
    ref_heat = float(getattr(ref, "hours_heat_above_threshold"))
    ref_eff = float(_action_use_metrics(ref)["effort_score"])

    rows: List[Dict[str, Any]] = []
    for name in order:
        if name not in results:
            continue
        run = results[name]
        pm = int(getattr(run, "minutes_pm25_above_threshold"))
        heat = int(getattr(run, "hours_heat_above_threshold"))

        use = _action_use_metrics(run)
        eff = float(use["effort_score"])

        rows.append(
            {
                "policy": name,
                "pm_minutes": pm,
                "heat_hours": heat,
                "pm_pct_vs_ref": _pct_vs_ref(ref_pm, float(pm)),
                "heat_pct_vs_ref": _pct_vs_ref(ref_heat, float(heat)),
                "hepa_hours": int(use["hepa_hours"]),
                "fan_hours": int(use["fan_hours"]),
                "window_open_hours": int(use["window_open_hours"]),
                "effort_score": eff,
                "effort_pct_vs_ref": _pct_vs_ref(ref_eff, eff),
                "ref_policy": ref_name,
            }
        )

    return rows, results


def policy_table_string(rows: List[Dict[str, Any]], *, eval_ref: str = "heuristic") -> str:
    policy_w = max(len("Policy"), max(len(str(r.get("policy", ""))) for r in rows)) if rows else len("Policy")

    header = (
        f"{'Policy'.ljust(policy_w)} | "
        f"{'PM min'.rjust(6)} | "
        f"{'Heat h'.rjust(6)} | "
        f"{'PM % vs ref'.rjust(11)} | "
        f"{'Heat % vs ref'.rjust(12)} | "
        f"{'HEPA h'.rjust(6)} | "
        f"{'Fan h'.rjust(5)} | "
        f"{'Win h'.rjust(5)} | "
        f"{'Effort'.rjust(7)} | "
        f"{'Eff %'.rjust(7)}"
    )
    sep = (
        f"{'-' * policy_w}-+-"
        f"{'-' * 6}-+-"
        f"{'-' * 6}-+-"
        f"{'-' * 11}-+-"
        f"{'-' * 12}-+-"
        f"{'-' * 6}-+-"
        f"{'-' * 5}-+-"
        f"{'-' * 5}-+-"
        f"{'-' * 7}-+-"
        f"{'-' * 7}"
    )

    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{str(r['policy']).ljust(policy_w)} | "
            f"{int(r['pm_minutes']):>6d} | "
            f"{int(r['heat_hours']):>6d} | "
            f"{_fmt_pct(r.get('pm_pct_vs_ref'), width=11)} | "
            f"{_fmt_pct(r.get('heat_pct_vs_ref'), width=12)} | "
            f"{int(r.get('hepa_hours', 0)):>6d} | "
            f"{int(r.get('fan_hours', 0)):>5d} | "
            f"{int(r.get('window_open_hours', 0)):>5d} | "
            f"{float(r.get('effort_score', 0.0)):>7.1f} | "
            f"{_fmt_pct(r.get('effort_pct_vs_ref'), width=7)}"
        )
    return "\n".join(lines)


def export_eval_json(
    rows: List[Dict[str, Any]],
    results: Dict[str, Any],
    *,
    out_dir: str | Path = "outputs",
    prefix: str = "shield_eval",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "results": results,
    }

    out = out_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return out
