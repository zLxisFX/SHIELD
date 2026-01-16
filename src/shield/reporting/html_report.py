"""
SHIELD HTML Report Generator (offline, no dependencies).

Writes a single self-contained HTML report that:
- summarizes key run metadata
- shows policy comparison table (rows from evaluate_policies)
- highlights robust vs reference (heuristic) if present
- optionally renders an hour-by-hour schedule if plan actions are available in results

Designed to be PyInstaller-friendly and resilient to schema drift.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import html


# -----------------------
# Small helpers
# -----------------------

def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _escape(x: Any) -> str:
    if x is None:
        return ""
    try:
        return html.escape(str(x))
    except Exception:
        return html.escape(repr(x))


def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _fmt(x: Any, *, digits: int = 2) -> str:
    v = _num(x)
    if v is None:
        return _escape(x)
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return f"{v:.{digits}f}"


def _first_key(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _get_metric(row: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    v = _first_key(row, keys)
    return default if v is None else v


def _safe_filename(s: str) -> str:
    # conservative: keep alnum, dash, underscore; replace others with '_'
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "report"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


# -----------------------
# Plan extraction (optional)
# -----------------------

def _normalize_action_item(item: Any) -> Optional[Dict[str, Any]]:
    """
    Try to normalize an action/hour item into:
      {"t": "...", "windows_open": bool, "hepa_on": bool, "fan_on": bool, "notes": "..."}

    Supports dicts, dataclasses, and objects with attributes.
    """
    if item is None:
        return None

    if is_dataclass(item):
        item = asdict(item)

    if isinstance(item, dict):
        t = item.get("t") or item.get("time") or item.get("datetime")
        windows_open = item.get("windows_open")
        hepa_on = item.get("hepa_on")
        fan_on = item.get("fan_on")
        notes = item.get("notes")
    else:
        # object-ish
        t = getattr(item, "t", None) or getattr(item, "time", None) or getattr(item, "datetime", None)
        windows_open = getattr(item, "windows_open", None)
        hepa_on = getattr(item, "hepa_on", None)
        fan_on = getattr(item, "fan_on", None)
        notes = getattr(item, "notes", None)

    # format time
    if isinstance(t, datetime):
        t_str = t.replace(microsecond=0).isoformat()
    elif t is None:
        t_str = ""
    else:
        t_str = str(t)

    # normalize notes
    if isinstance(notes, (list, tuple)):
        notes_str = "; ".join(str(x) for x in notes if x is not None)
    elif notes is None:
        notes_str = ""
    else:
        notes_str = str(notes)

    # require at least the booleans to exist
    if windows_open is None and hepa_on is None and fan_on is None:
        return None

    return {
        "t": t_str,
        "windows_open": bool(windows_open) if windows_open is not None else False,
        "hepa_on": bool(hepa_on) if hepa_on is not None else False,
        "fan_on": bool(fan_on) if fan_on is not None else False,
        "notes": notes_str,
    }


def _extract_plan_from_results(results: Dict[str, Any], policy: str) -> Optional[List[Dict[str, Any]]]:
    """
    Best-effort extraction. Accepts several possible shapes:

    results[policy] might be:
      - {"plan": [ActionsHour...], ...}
      - {"actions": [...], ...}
      - {"schedule": [...], ...}
      - directly a list of ActionsHour/dicts
    """
    if not isinstance(results, dict):
        return None

    payload = results.get(policy)
    if payload is None:
        return None

    # direct list
    if isinstance(payload, list):
        out: List[Dict[str, Any]] = []
        for it in payload:
            n = _normalize_action_item(it)
            if n is not None:
                out.append(n)
        return out or None

    # dict with candidate keys
    if isinstance(payload, dict):
        for key in ("plan", "actions", "schedule", "hourly", "hours"):
            if key in payload and isinstance(payload[key], list):
                out2: List[Dict[str, Any]] = []
                for it in payload[key]:
                    n = _normalize_action_item(it)
                    if n is not None:
                        out2.append(n)
                return out2 or None

    return None


# -----------------------
# HTML rendering
# -----------------------

def _preferred_columns(rows: List[Dict[str, Any]]) -> List[str]:
    # Try newer names first, but keep backward compat.
    preferred = [
        "policy",
        "pm_minutes", "pm_over_min",
        "pm_pct_vs_ref",
        "heat_hours", "heat_over_h",
        "heat_pct_vs_ref",
        "hepa_hours",
        "fan_hours",
        "window_open_hours",
        "effort_score",
        "effort_pct_vs_ref",
        "budget_ok",
        "ref_policy",
    ]

    keys = sorted({k for r in rows for k in r.keys()})
    cols: List[str] = []

    for k in preferred:
        if k in keys and k not in cols:
            cols.append(k)

    for k in keys:
        if k not in cols:
            cols.append(k)

    return cols


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = {"robust": 0, "heuristic": 1}
    def key(r: Dict[str, Any]) -> Tuple[int, str]:
        p = str(r.get("policy", ""))
        return (order.get(p, 50), p)
    return sorted(rows, key=key)


def _card(title: str, items: List[Tuple[str, Any]]) -> str:
    li = []
    for k, v in items:
        li.append(f"<div class='kv'><div class='k'>{_escape(k)}</div><div class='v'>{_escape(_fmt(v))}</div></div>")
    return f"""
      <div class="card">
        <div class="card-title">{_escape(title)}</div>
        <div class="card-body">
          {''.join(li)}
        </div>
      </div>
    """


def _render_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>(no rows)</p>"

    cols = _preferred_columns(rows)

    # header
    th = "".join(f"<th>{_escape(c)}</th>" for c in cols)

    # body
    trs = []
    for r in _sort_rows(rows):
        tds = []
        for c in cols:
            v = r.get(c, "")
            cls = ""
            if c in ("pm_pct_vs_ref", "heat_pct_vs_ref", "effort_pct_vs_ref"):
                vv = _num(v)
                if vv is not None:
                    # Positive % means improvement (wins)
                    cls = "pos" if vv > 0 else ("neg" if vv < 0 else "zero")
            if c in ("pm_minutes", "pm_over_min", "heat_hours", "heat_over_h"):
                cls = cls or "mono"
            tds.append(f"<td class='{cls}'>{_escape(_fmt(v))}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")

    return f"""
      <div class="table-wrap">
        <table>
          <thead><tr>{th}</tr></thead>
          <tbody>
            {''.join(trs)}
          </tbody>
        </table>
      </div>
    """


def _render_schedule(plan: List[Dict[str, Any]]) -> str:
    if not plan:
        return "<p class='muted'>(no schedule available)</p>"

    # Group by day (based on ISO date prefix if possible)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for it in plan:
        t = it.get("t", "")
        day = t[:10] if isinstance(t, str) and len(t) >= 10 else "schedule"
        grouped.setdefault(day, []).append(it)

    blocks = []
    for day in sorted(grouped.keys()):
        rows = grouped[day]
        # keep order as given
        trs = []
        for it in rows:
            t = it.get("t", "")
            time = t[11:16] if isinstance(t, str) and len(t) >= 16 else t
            w = "OPEN" if it.get("windows_open") else "CLOSED"
            h = "ON" if it.get("hepa_on") else "OFF"
            f = "ON" if it.get("fan_on") else "OFF"
            notes = it.get("notes", "")
            trs.append(
                "<tr>"
                f"<td class='mono'>{_escape(time)}</td>"
                f"<td>{_escape(w)}</td>"
                f"<td>{_escape(h)}</td>"
                f"<td>{_escape(f)}</td>"
                f"<td>{_escape(notes)}</td>"
                "</tr>"
            )

        blocks.append(f"""
          <h3 class="h3">{_escape(day)}</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th><th>Windows</th><th>HEPA</th><th>Fan</th><th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {''.join(trs)}
              </tbody>
            </table>
          </div>
        """)

    return "".join(blocks)


def write_html_report(
    path: Path,
    *,
    rows: List[Dict[str, Any]],
    results: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    focus_policy: str = "robust",
    ref_policy: str = "heuristic",
) -> Path:
    """
    Write an HTML report to `path`. Returns the written path.

    - rows: from evaluate_policies()
    - results: from evaluate_policies()
    - meta: run metadata (scenario, archetype, etc.)
    """
    _ensure_parent(path)

    meta = dict(meta or {})
    meta.setdefault("generated_at", _now_iso())
    meta.setdefault("focus_policy", focus_policy)
    meta.setdefault("ref_policy", ref_policy)

    # lookup key metrics
    by_pol = {str(r.get("policy", "")): r for r in rows}

    def pack_card(pol: str) -> List[Tuple[str, Any]]:
        r = by_pol.get(pol, {})
        return [
            ("PM exceedance (min)", _get_metric(r, ["pm_minutes", "pm_over_min"], "")),
            ("Heat ≥ threshold (h)", _get_metric(r, ["heat_hours", "heat_over_h"], "")),
            ("HEPA hours", _get_metric(r, ["hepa_hours"], "")),
            ("Fan hours", _get_metric(r, ["fan_hours"], "")),
            ("Windows open (h)", _get_metric(r, ["window_open_hours"], "")),
            ("Effort score", _get_metric(r, ["effort_score"], "")),
            ("PM improvement vs ref (%)", _get_metric(r, ["pm_pct_vs_ref"], "")),
            ("Heat improvement vs ref (%)", _get_metric(r, ["heat_pct_vs_ref"], "")),
            ("Effort reduction vs ref (%)", _get_metric(r, ["effort_pct_vs_ref"], "")),
        ]

    focus_items = pack_card(focus_policy)
    ref_items = pack_card(ref_policy)

    # optional schedule
    schedule = _extract_plan_from_results(results, focus_policy)

    # meta table
    meta_rows = []
    for k in sorted(meta.keys()):
        meta_rows.append(f"<tr><td class='mono'>{_escape(k)}</td><td>{_escape(meta[k])}</td></tr>")
    meta_table = f"""
      <div class="table-wrap">
        <table>
          <thead><tr><th>Field</th><th>Value</th></tr></thead>
          <tbody>{''.join(meta_rows)}</tbody>
        </table>
      </div>
    """

    title_bits = []
    if "scenario" in meta:
        title_bits.append(str(meta["scenario"]))
    if "archetype" in meta:
        title_bits.append(str(meta["archetype"]))
    title = " / ".join(title_bits) if title_bits else "SHIELD Report"

    css = """
    :root { --bg:#0b1020; --panel:#121a33; --text:#e9ecf1; --muted:#a9b1c6; --line:#26325e; }
    body { margin:0; padding:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Arial; }
    .wrap { max-width:1100px; margin:0 auto; padding:28px 18px 60px; }
    .title { font-size:28px; font-weight:800; margin:0 0 6px; }
    .sub { color:var(--muted); margin:0 0 18px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 14px 10px; }
    .card-title { font-weight:800; margin:0 0 10px; font-size:15px; }
    .card-body { display:grid; gap:8px; }
    .kv { display:flex; justify-content:space-between; gap:16px; border-top:1px dashed rgba(255,255,255,.07); padding-top:8px; }
    .kv:first-child { border-top:none; padding-top:0; }
    .k { color:var(--muted); }
    .v { font-weight:700; }
    .h2 { font-size:18px; font-weight:850; margin:22px 0 10px; }
    .h3 { font-size:15px; font-weight:850; margin:16px 0 8px; color:#dfe6ff; }
    .muted { color:var(--muted); }
    .table-wrap { overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:14px; }
    table { width:100%; border-collapse:separate; border-spacing:0; min-width:720px; }
    thead th { position:sticky; top:0; background:#141f3d; z-index:1; }
    th, td { padding:10px 10px; border-bottom:1px solid rgba(255,255,255,.06); vertical-align:top; }
    th { text-align:left; font-weight:850; }
    tr:last-child td { border-bottom:none; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
    .pos { color:#8dffb5; font-weight:800; }
    .neg { color:#ff9aa2; font-weight:800; }
    .zero { color:#e9ecf1; font-weight:800; }
    .footer { margin-top:22px; color:var(--muted); font-size:12px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } table { min-width: 900px; } }
    """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">SHIELD Mode Report</h1>
    <p class="sub">{_escape(title)} • generated { _escape(meta.get("generated_at","")) }</p>

    <div class="h2">Run metadata</div>
    {meta_table}

    <div class="h2">Key outcomes</div>
    <div class="grid">
      {_card(focus_policy, focus_items)}
      {_card(ref_policy, ref_items)}
    </div>

    <div class="h2">Policy comparison table</div>
    {_render_table(rows)}

    <div class="h2">Schedule (if available): { _escape(focus_policy) }</div>
    {_render_schedule(schedule) if schedule else "<p class='muted'>(Schedule not embedded in results; report still includes the full comparison table.)</p>"}

    <div class="footer">
      SHIELD • offline report • this file is self-contained (no external assets).
    </div>
  </div>
</body>
</html>
"""

    path.write_text(html_doc, encoding="utf-8")
    return path
