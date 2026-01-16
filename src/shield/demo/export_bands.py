"""
Export utilities for ensemble uncertainty bands.

Outputs:
- JSON file with:
  - metadata (n, seed, thresholds)
  - per-hour quantile bands for PM and temperature

Also provides a lightweight "confidence meter" score derived from band widths.
(For now: heuristic; later: calibrated from coverage on held-out events.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from shield.uq.ensemble import EnsembleResult


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="minutes")


def _jsonable(obj: Any) -> Any:
    """
    Convert dataclasses + datetimes into JSON-safe objects.
    """
    if isinstance(obj, datetime):
        return _iso(obj)

    if is_dataclass(obj):
        return _jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]

    return obj


def confidence_meter_from_bands(
    ens: EnsembleResult,
    *,
    pm_scale: float = 25.0,
    t_scale: float = 6.0,
) -> Dict[str, float]:
    """
    Heuristic confidence meter in [0, 1] derived from average 95% band widths.

    Interpretation:
    - Narrow bands => higher confidence
    - Wide bands   => lower confidence

    pm_scale and t_scale set the "width where confidence ~ e^-1".
    (Later we calibrate these based on real coverage and decision errors.)
    """
    if not ens.bands:
        return {"pm_conf": 0.0, "t_conf": 0.0}

    pm_widths = [max(0.0, b.pm97_5 - b.pm02_5) for b in ens.bands]
    t_widths = [max(0.0, b.t97_5 - b.t02_5) for b in ens.bands]

    pm_w = sum(pm_widths) / len(pm_widths)
    t_w = sum(t_widths) / len(t_widths)

    # exp decay to map width -> [0,1]
    # clamp to be safe
    import math
    pm_conf = float(max(0.0, min(1.0, math.exp(-pm_w / max(1e-9, pm_scale)))))
    t_conf = float(max(0.0, min(1.0, math.exp(-t_w / max(1e-9, t_scale)))))

    return {"pm_conf": pm_conf, "t_conf": t_conf}


def export_ensemble_bands(
    ens: EnsembleResult,
    out_dir: Path | str = "outputs",
    prefix: str = "shield_bands",
) -> Path:
    """
    Export ensemble bands to a JSON file.
    Returns path to the JSON file.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{prefix}_{ts}.json"

    conf = confidence_meter_from_bands(ens)

    payload: Dict[str, Any] = {
        "meta": {
            "created_at": _iso(datetime.now()),
            "n": ens.n,
            "seed": ens.seed,
            "pm_threshold": ens.pm_threshold,
            "heat_threshold_c": ens.heat_threshold_c,
            "confidence_meter": conf,
            "notes": [
                "Bands are from parameter-sampled ensemble runs (priors).",
                "This is Offline Demo UQ; later we calibrate/condition on sensors.",
            ],
        },
        "bands": _jsonable(ens.bands),
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def preview_bands_next_hours(
    ens: EnsembleResult,
    *,
    hours: int = 6,
) -> List[str]:
    """
    Handy console preview lines:
    - shows median + 80% range for PM and temperature
    """
    lines: List[str] = []
    n = min(int(hours), len(ens.bands))
    for i in range(n):
        b = ens.bands[i]
        lines.append(
            f"{_iso(b.t)} | "
            f"PM50={b.pm50:5.1f} (80% {b.pm10:5.1f}..{b.pm90:5.1f}) | "
            f"T50={b.t50:4.1f} (80% {b.t10:4.1f}..{b.t90:4.1f})"
        )
    return lines
