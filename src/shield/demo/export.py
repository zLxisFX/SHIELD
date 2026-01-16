"""
Export utilities for SHIELD demo runs.

Fixes:
- Avoid overwriting exports when two runs occur in the same second
  by using microsecond timestamps + a uniqueness fallback.
- JSON serialization handles datetime safely.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def _iso(dt: Any) -> Any:
    if isinstance(dt, datetime):
        return dt.isoformat(timespec="seconds")
    return dt


def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert dataclasses + datetimes to JSON-safe structures.
    """
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")

    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    return obj


def _pick_notes(action: Any) -> str:
    """
    Extract notes from an Action-like object or dict.
    """
    if isinstance(action, dict):
        notes = action.get("notes", [])
    else:
        notes = getattr(action, "notes", [])
    if not notes:
        return ""
    if isinstance(notes, (list, tuple)):
        return ", ".join(str(x) for x in notes if x is not None)
    return str(notes)


def _unique_stem(prefix: str) -> str:
    # microsecond-level timestamp to prevent same-second overwrites
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}"


def _reserve_unique_paths(out_dir: Path, prefix: str) -> Tuple[Path, Path]:
    """
    Create a unique stem and ensure both .txt and .json do not already exist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _unique_stem(prefix)
    txt_path = out_dir / f"{stem}.txt"
    json_path = out_dir / f"{stem}.json"

    # Extremely defensive: if a collision somehow happens, add a counter
    if txt_path.exists() or json_path.exists():
        for i in range(1, 1000):
            stem_i = f"{stem}_{i}"
            txt_i = out_dir / f"{stem_i}.txt"
            json_i = out_dir / f"{stem_i}.json"
            if (not txt_i.exists()) and (not json_i.exists()):
                return txt_i, json_i

    return txt_path, json_path


def _format_txt(run: Any) -> str:
    """
    Human-friendly TXT plan export.
    Works with dataclass RunResult/SimulationRun or dict-like equivalents.
    """
    # Basic header fields (best-effort)
    pm_thr = getattr(run, "pm25_threshold", None) if not isinstance(run, dict) else run.get("pm25_threshold")
    ht_thr = getattr(run, "heat_threshold_c", None) if not isinstance(run, dict) else run.get("heat_threshold_c")
    pm_min = getattr(run, "minutes_pm25_above_threshold", None) if not isinstance(run, dict) else run.get("minutes_pm25_above_threshold")
    ht_hr = getattr(run, "hours_heat_above_threshold", None) if not isinstance(run, dict) else run.get("hours_heat_above_threshold")

    outputs = getattr(run, "outputs", []) if not isinstance(run, dict) else run.get("outputs", [])
    actions = getattr(run, "actions", []) if not isinstance(run, dict) else run.get("actions", [])

    lines = []
    lines.append("SHIELD Plan Export")
    if pm_thr is not None:
        lines.append(f"- PM threshold (µg/m³): {pm_thr}")
    if ht_thr is not None:
        lines.append(f"- Heat threshold (°C): {ht_thr}")
    if pm_min is not None:
        lines.append(f"- Minutes above PM threshold: {pm_min}")
    if ht_hr is not None:
        lines.append(f"- Hours above heat threshold: {ht_hr}")
    lines.append("")
    lines.append("Hour-by-hour:")
    lines.append("t | PM_in | T_in | notes")
    lines.append("-" * 72)

    n = min(len(outputs), len(actions))
    for i in range(n):
        o = outputs[i]
        a = actions[i]

        if isinstance(o, dict):
            t = o.get("t")
            pm_in = o.get("pm25_in")
            t_in = o.get("temp_in_c")
        else:
            t = getattr(o, "t", None)
            pm_in = getattr(o, "pm25_in", None)
            t_in = getattr(o, "temp_in_c", None)

        t_str = _iso(t)
        notes = _pick_notes(a)

        pm_str = f"{pm_in:5.1f}" if isinstance(pm_in, (int, float)) else "  n/a"
        ti_str = f"{t_in:4.1f}" if isinstance(t_in, (int, float)) else " n/a"

        lines.append(f"{t_str} | {pm_str} | {ti_str} | {notes}")

    return "\n".join(lines) + "\n"


def export_runresult(run: Any, out_dir: Path | str = Path("outputs"), prefix: str = "shield_plan") -> Tuple[Path, Path]:
    """
    Export a run result to:
      - TXT plan (human readable)
      - JSON payload (machine readable)

    Returns: (txt_path, json_path)
    """
    out_dir = Path(out_dir)

    txt_path, json_path = _reserve_unique_paths(out_dir, prefix)

    # TXT
    txt_path.write_text(_format_txt(run), encoding="utf-8")

    # JSON
    payload: Dict[str, Any] = _to_jsonable(run)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return txt_path, json_path
