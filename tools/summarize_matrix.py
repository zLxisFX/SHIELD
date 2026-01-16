from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s.lower() in {"none", "null", "nan"}:
        return None
    if s.lower() in {"unlimited", "inf", "infinite"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _get(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return str(row[k]).strip()
    return None


def _get_num(row: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    return _safe_float(_get(row, keys))


def _latest_match(pattern: str) -> Path:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    matches = sorted(matches, key=lambda p: os.path.getmtime(p))
    return Path(matches[-1])


def _resolve_input(path_or_glob: str) -> Path:
    s = str(path_or_glob)
    if any(ch in s for ch in ["*", "?", "["]):
        return _latest_match(s)
    return Path(s)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


@dataclass(frozen=True)
class RunKey:
    scenario: str
    archetype: str
    area: str
    start_hour: str
    hepa_budget: str


@dataclass
class PairSummary:
    key: RunKey
    pm_ref: Optional[float]
    pm_robust: Optional[float]
    heat_ref: Optional[float]
    heat_robust: Optional[float]
    effort_ref: Optional[float]
    effort_robust: Optional[float]
    hepa_h_ref: Optional[float]
    hepa_h_robust: Optional[float]
    fan_h_ref: Optional[float]
    fan_h_robust: Optional[float]
    win_pm: Optional[bool]
    win_heat: Optional[bool]
    win_effort: Optional[bool]
    pm_pct: Optional[float]
    heat_pct: Optional[float]
    effort_pct: Optional[float]
    budget_ok: Optional[bool]


def _pct_improve(ref: Optional[float], new: Optional[float]) -> Optional[float]:
    if ref is None or new is None:
        return None
    if ref == 0:
        return None
    return 100.0 * (ref - new) / ref


def _fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def _summ_stats(vals: List[float]) -> Tuple[float, float, float]:
    vals_sorted = sorted(vals)
    med = statistics.median(vals_sorted)
    return (min(vals_sorted), med, max(vals_sorted))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path_or_glob", type=str, help="e.g. outputs\\eval_matrix_*.csv (we pick latest match)")
    ap.add_argument("--ref", type=str, default="heuristic", help="reference policy name")
    ap.add_argument("--policy", type=str, default="robust", help="policy to compare vs ref")
    ap.add_argument("--horizon-hours", type=int, default=72, help="used for HEPA budget check")
    ap.add_argument("--outdir", type=str, default="outputs", help="where to write summary markdown")
    args = ap.parse_args()

    path = _resolve_input(args.csv_path_or_glob)
    rows = _read_csv(path)

    if not rows:
        print(f"[error] CSV has no rows: {path}")
        return 2

    # column guesses
    k_scenario = ["scenario", "scenario_name"]
    k_archetype = ["archetype"]
    k_area = ["area", "floor_area_m2", "area_m2"]
    k_start = ["start_hour", "start"]
    k_budget = ["hepa_budget_h_per_day", "hepa_budget", "budget"]

    k_policy = ["policy"]
    k_pm = ["pm_minutes", "pm_over_min", "pm_over_minutes"]
    k_heat = ["heat_hours", "heat_over_h", "heat_over_hours"]
    k_effort = ["effort_score", "effort"]
    k_hepa_h = ["hepa_hours", "hepa_h"]
    k_fan_h = ["fan_hours", "fan_h"]

    # group rows by run-key
    grouped: Dict[RunKey, Dict[str, Dict[str, Any]]] = {}

    for r in rows:
        scenario = _get(r, k_scenario) or "?"
        archetype = _get(r, k_archetype) or "?"
        area = _get(r, k_area) or "?"
        start_hour = _get(r, k_start) or "?"
        budget_raw = _get(r, k_budget)

        # normalize budget display: None/blank => unlimited
        b = _safe_float(budget_raw)
        budget_str = "unlimited" if b is None else f"{b:g}"

        pol = (_get(r, k_policy) or "").strip()
        if not pol:
            continue

        key = RunKey(
            scenario=str(scenario),
            archetype=str(archetype),
            area=str(area),
            start_hour=str(start_hour),
            hepa_budget=str(budget_str),
        )
        if key not in grouped:
            grouped[key] = {}
        grouped[key][pol] = r

    if not grouped:
        print("[error] Could not form any run groups (missing columns like scenario/archetype/policy).")
        print("First row keys:", sorted(list(rows[0].keys())))
        return 2

    # build summaries for ref vs policy
    ref_name = str(args.ref)
    pol_name = str(args.policy)

    days = max(1.0, float(args.horizon_hours) / 24.0)

    pairs: List[PairSummary] = []
    skipped = 0

    for key, by_pol in grouped.items():
        if ref_name not in by_pol or pol_name not in by_pol:
            skipped += 1
            continue

        rr = by_pol[ref_name]
        pr = by_pol[pol_name]

        pm_ref = _get_num(rr, k_pm)
        pm_pol = _get_num(pr, k_pm)

        heat_ref = _get_num(rr, k_heat)
        heat_pol = _get_num(pr, k_heat)

        effort_ref = _get_num(rr, k_effort)
        effort_pol = _get_num(pr, k_effort)

        hepa_ref = _get_num(rr, k_hepa_h)
        hepa_pol = _get_num(pr, k_hepa_h)

        fan_ref = _get_num(rr, k_fan_h)
        fan_pol = _get_num(pr, k_fan_h)

        pm_pct = _pct_improve(pm_ref, pm_pol)
        heat_pct = _pct_improve(heat_ref, heat_pol)
        effort_pct = _pct_improve(effort_ref, effort_pol)

        win_pm = None if pm_pct is None else (pm_pct > 0)
        win_heat = None if heat_pct is None else (heat_pct > 0)
        win_effort = None if effort_pct is None else (effort_pct > 0)

        # budget check for the compared policy
        b = _safe_float(key.hepa_budget)
        budget_ok: Optional[bool]
        if b is None:
            budget_ok = True
        else:
            # b is hours/day; total allowed over horizon is b*days
            allowed = float(b) * float(days)
            if hepa_pol is None:
                budget_ok = None
            else:
                budget_ok = (float(hepa_pol) <= allowed + 1e-9)

        pairs.append(
            PairSummary(
                key=key,
                pm_ref=pm_ref,
                pm_robust=pm_pol,
                heat_ref=heat_ref,
                heat_robust=heat_pol,
                effort_ref=effort_ref,
                effort_robust=effort_pol,
                hepa_h_ref=hepa_ref,
                hepa_h_robust=hepa_pol,
                fan_h_ref=fan_ref,
                fan_h_robust=fan_pol,
                win_pm=win_pm,
                win_heat=win_heat,
                win_effort=win_effort,
                pm_pct=pm_pct,
                heat_pct=heat_pct,
                effort_pct=effort_pct,
                budget_ok=budget_ok,
            )
        )

    if not pairs:
        print(f"[error] No comparable runs found for ref='{ref_name}' and policy='{pol_name}'.")
        print(f"Grouped runs: {len(grouped)} | skipped (missing either row): {skipped}")
        return 2

    # aggregate stats
    pm_vals = [p.pm_pct for p in pairs if p.pm_pct is not None]
    heat_vals = [p.heat_pct for p in pairs if p.heat_pct is not None]
    effort_vals = [p.effort_pct for p in pairs if p.effort_pct is not None]

    def winrate(flag_name: str) -> float:
        flags = [getattr(p, flag_name) for p in pairs if getattr(p, flag_name) is not None]
        if not flags:
            return float("nan")
        return 100.0 * (sum(1 for x in flags if x) / len(flags))

    pm_win = winrate("win_pm")
    heat_win = winrate("win_heat")
    effort_win = winrate("win_effort")
    both_win_flags = [
        (p.win_pm and p.win_heat)
        for p in pairs
        if (p.win_pm is not None and p.win_heat is not None)
    ]
    both_win = 100.0 * (sum(1 for x in both_win_flags if x) / len(both_win_flags)) if both_win_flags else float("nan")

    budget_flags = [p.budget_ok for p in pairs if p.budget_ok is not None]
    budget_ok_rate = 100.0 * (sum(1 for x in budget_flags if x) / len(budget_flags)) if budget_flags else float("nan")

    def rng_med(vals: List[float]) -> str:
        if not vals:
            return ""
        lo, med, hi = _summ_stats(vals)
        return f"{med:.2f}% (range {lo:.2f}% to {hi:.2f}%)"

    # print compact table
    print(f"\nSummary for: {path}")
    print(f"Compared: {pol_name} vs {ref_name} | runs={len(pairs)} | horizon={args.horizon_hours}h\n")
    header = ["scenario", "start", "budget", "pm_%", "heat_%", "effort_%", "hepa_h(pol)", "fan_h(pol)", "budget_ok"]
    print(" | ".join(header))
    print("-+-".join(["-" * len(h) for h in header]))
    for p in sorted(pairs, key=lambda x: (x.key.scenario, x.key.start_hour, x.key.hepa_budget)):
        print(
            f"{p.key.scenario} | {p.key.start_hour} | {p.key.hepa_budget} | "
            f"{_fmt(p.pm_pct, 2)} | {_fmt(p.heat_pct, 2)} | {_fmt(p.effort_pct, 2)} | "
            f"{_fmt(p.hepa_h_robust, 0)} | {_fmt(p.fan_h_robust, 0)} | {p.budget_ok}"
        )

    print("\nAggregate (median + range):")
    if pm_vals:
        print(" - PM minutes improvement:", rng_med(pm_vals))
    if heat_vals:
        print(" - Heat hours improvement:", rng_med(heat_vals))
    if effort_vals:
        print(" - Effort reduction:", rng_med(effort_vals))

    print("\nWin rates:")
    print(f" - PM wins: {pm_win:.1f}%")
    print(f" - Heat wins: {heat_win:.1f}%")
    print(f" - Both (PM & Heat) wins: {both_win:.1f}%")
    print(f" - Effort wins: {effort_win:.1f}%")
    print(f" - Budget respected: {budget_ok_rate:.1f}%")

    # write proposal-ready markdown
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / f"summary_matrix_{pol_name}_vs_{ref_name}_{_now_stamp()}.md"

    md = []
    md.append(f"# SHIELD Eval Matrix Summary\n")
    md.append(f"- Input: `{path}`\n")
    md.append(f"- Compared: **{pol_name} vs {ref_name}**\n")
    md.append(f"- Runs: **{len(pairs)}** (horizon={args.horizon_hours}h)\n")
    md.append("\n## Headline metrics (median, range)\n")
    if pm_vals:
        md.append(f"- **PM exceedance minutes**: {rng_med(pm_vals)}\n")
    if heat_vals:
        md.append(f"- **Heat hours ≥ threshold**: {rng_med(heat_vals)}\n")
    if effort_vals:
        md.append(f"- **Effort score**: {rng_med(effort_vals)}\n")
    md.append("\n## Win rates\n")
    md.append(f"- PM wins: **{pm_win:.1f}%**\n")
    md.append(f"- Heat wins: **{heat_win:.1f}%**\n")
    md.append(f"- Both wins (PM & Heat): **{both_win:.1f}%**\n")
    md.append(f"- Effort wins: **{effort_win:.1f}%**\n")
    md.append(f"- Budget respected: **{budget_ok_rate:.1f}%**\n")

    md.append("\n## Per-run details\n")
    md.append("| scenario | start | budget | pm_% | heat_% | effort_% | hepa_h(pol) | fan_h(pol) | budget_ok |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for p in sorted(pairs, key=lambda x: (x.key.scenario, x.key.start_hour, x.key.hepa_budget)):
        md.append(
            f"| {p.key.scenario} | {p.key.start_hour} | {p.key.hepa_budget} | "
            f"{_fmt(p.pm_pct,2)} | {_fmt(p.heat_pct,2)} | {_fmt(p.effort_pct,2)} | "
            f"{_fmt(p.hepa_h_robust,0)} | {_fmt(p.fan_h_robust,0)} | {p.budget_ok} |\n"
        )

    md_path.write_text("".join(md), encoding="utf-8")
    print(f"\nWrote proposal-ready summary:\n- {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
