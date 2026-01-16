from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from glob import glob


@dataclass(frozen=True)
class PlanRow:
    t: datetime
    pm_in: float
    t_in: float
    notes: str
    windows_open: bool
    hepa_on: bool
    fan_on: bool


def _parse_iso_dt(s: str) -> datetime:
    # Handles "2026-01-15T00:00:00" etc.
    return datetime.fromisoformat(s.strip())


def _infer_actions_from_notes(notes: str) -> Tuple[bool, bool, bool]:
    """
    Infer action state from notes string, respecting overrides.
    """
    s = notes or ""

    # requested actions (from positive phrases found in exports)
    windows_req = ("windows OPEN" in s) or ("Ventilate (windows OPEN)" in s)
    hepa_req = ("Run HEPA / filter" in s)
    fan_req = ("Run fan" in s)

    # overrides win
    windows_open = windows_req and ("Override: windows CLOSED" not in s)
    hepa_on = hepa_req and ("Override: HEPA OFF" not in s)
    fan_on = fan_req and ("Override: fan OFF" not in s)

    return bool(windows_open), bool(hepa_on), bool(fan_on)


def read_plan_txt(path: Path) -> List[PlanRow]:
    text = path.read_text(encoding="utf-8")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    # Find the table start: line after a row of dashes or after "Hour-by-hour:"
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("t | pm_in"):
            start_idx = i + 1
            break
    if start_idx is None:
        # fallback: find "Hour-by-hour:" then skip next 2 lines
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith("hour-by-hour"):
                start_idx = i + 3
                break
    if start_idx is None:
        raise ValueError(f"Could not find hour-by-hour table in {path}")

    rows: List[PlanRow] = []
    for ln in lines[start_idx:]:
        if not ln.strip():
            continue
        if "|" not in ln:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 4:
            continue

        try:
            t = _parse_iso_dt(parts[0])
            pm_in = float(parts[1])
            t_in = float(parts[2])
            notes = parts[3]
        except Exception:
            continue

        w, h, f = _infer_actions_from_notes(notes)
        rows.append(PlanRow(t=t, pm_in=pm_in, t_in=t_in, notes=notes, windows_open=w, hepa_on=h, fan_on=f))

    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    return rows


def _index_by_time(rows: List[PlanRow]) -> Dict[datetime, PlanRow]:
    return {r.t: r for r in rows}


def _hour_filter(t: datetime, h0: Optional[int], h1: Optional[int]) -> bool:
    if h0 is None or h1 is None:
        return True
    # inclusive start, exclusive end (wrap not supported)
    return h0 <= t.hour < h1


def _resolve_path(pat: str) -> Path:
    # Expand wildcards like outputs\shield_plan_robust_*.txt (Windows cmd won't do this for us)
    if any(ch in pat for ch in ["*", "?", "["]):
        matches = [Path(m) for m in glob(pat)]
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {pat}")
        # pick newest
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]
    return Path(pat)


def compare(robust_path: Path, heuristic_path: Path, h0: Optional[int], h1: Optional[int], max_lines: int) -> None:
    r_rows = read_plan_txt(robust_path)
    h_rows = read_plan_txt(heuristic_path)

    r_map = _index_by_time(r_rows)
    h_map = _index_by_time(h_rows)

    times = sorted(set(r_map.keys()) & set(h_map.keys()))
    if not times:
        raise ValueError("No overlapping timestamps between plans.")

    def summary(rows: List[PlanRow]) -> str:
        open_h = sum(1 for x in rows if x.windows_open)
        hepa_h = sum(1 for x in rows if x.hepa_on)
        fan_h = sum(1 for x in rows if x.fan_on)
        return f"hours: windows_open={open_h}, hepa_on={hepa_h}, fan_on={fan_h}, total={len(rows)}"

    print(f"\nROBUST:   {robust_path}")
    print(f"HEURISTIC:{heuristic_path}")
    print("\n--- ACTION SUMMARY (entire plan) ---")
    print("robust    :", summary(r_rows))
    print("heuristic :", summary(h_rows))

    print("\n--- DIFFERENCES (filtered) ---")
    shown = 0
    diff_count = 0

    for t in times:
        if not _hour_filter(t, h0, h1):
            continue

        rr = r_map[t]
        hh = h_map[t]

        # Compare actions only
        if (rr.windows_open, rr.hepa_on, rr.fan_on) != (hh.windows_open, hh.hepa_on, hh.fan_on):
            diff_count += 1
            if shown < max_lines:
                print(
                    f"{t.isoformat(timespec='minutes')} | "
                    f"robust(w,h,f)=({int(rr.windows_open)},{int(rr.hepa_on)},{int(rr.fan_on)}) "
                    f"heur(w,h,f)=({int(hh.windows_open)},{int(hh.hepa_on)},{int(hh.fan_on)}) | "
                    f"robust T={rr.t_in:4.1f},PM={rr.pm_in:5.1f} | heur T={hh.t_in:4.1f},PM={hh.pm_in:5.1f}"
                )
                print(f"  robust notes: {rr.notes}")
                print(f"  heur   notes: {hh.notes}")
                shown += 1

    print(f"\nDiff hours in filter: {diff_count} (printed {min(diff_count, max_lines)})")
    if h0 is not None and h1 is not None:
        print(f"Filter was hours [{h0}, {h1})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("robust_txt", type=str)
    ap.add_argument("heuristic_txt", type=str)
    ap.add_argument("--h0", type=int, default=None, help="filter start hour (0-23)")
    ap.add_argument("--h1", type=int, default=None, help="filter end hour (0-23), exclusive")
    ap.add_argument("--max", type=int, default=80, help="max diff lines to print")
    args = ap.parse_args()

    compare(_resolve_path(args.robust_txt), _resolve_path(args.heuristic_txt), args.h0, args.h1, args.max)


if __name__ == "__main__":
    main()