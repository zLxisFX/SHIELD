from __future__ import annotations

import csv
import json
import os
import sys
import threading
import traceback
import webbrowser
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk


# ----------------------------
# Small utilities (shared style)
# ----------------------------

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

    common = [
        "policy",
        "pm_minutes",
        "heat_hours",
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


def _pretty_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)\n"

    preferred = [
        "policy",
        "hepa_hours",
        "fan_hours",
        "window_open_hours",
        "effort_score",
        "effort_pct_vs_ref",
        "heat_hours",
        "heat_pct_vs_ref",
        "pm_minutes",
        "pm_pct_vs_ref",
        "ref_policy",
    ]
    cols = [c for c in preferred if any(c in r for r in rows)]
    extra = sorted({k for r in rows for k in r.keys() if k not in cols})
    cols += extra

    srows = [{c: _safe_str(r.get(c, "")) for c in cols} for r in rows]
    widths = {c: max(len(c), max(len(sr[c]) for sr in srows)) for c in cols}

    lines = []
    lines.append(" | ".join(c.ljust(widths[c]) for c in cols))
    lines.append("-+-".join("-" * widths[c] for c in cols))
    for sr in srows:
        lines.append(" | ".join(sr[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines) + "\n"


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_html_report(path: Path, *, title: str, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    table_txt = _pretty_table(rows)
    meta_lines = "\n".join(
        f"<li><b>{_html_escape(str(k))}</b>: {_html_escape(_safe_str(v))}</li>"
        for k, v in meta.items()
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{_html_escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 14px; }}
    pre {{ background: #f7f7f7; padding: 12px; border-radius: 10px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h2>{_html_escape(title)}</h2>

  <div class="card">
    <h3>Run details</h3>
    <ul>
      {meta_lines}
    </ul>
  </div>

  <div class="card">
    <h3>Results (table)</h3>
    <pre>{_html_escape(table_txt)}</pre>
  </div>

  <div class="card">
    <h3>Notes</h3>
    <p>This report is generated locally by SHIELD. CSV/JSON files are saved alongside this HTML.</p>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


# ----------------------------
# Core “run” function (same engine as CLI)
# ----------------------------

def run_one_eval(
    *,
    scenario: str,
    archetype: str,
    area_m2: float,
    start_hour: int,
    horizon_hours: int,
    occupants: int,
    hepa_budget_h_per_day: Optional[float],
    robust_kwargs: Dict[str, Any],
    include_baselines: bool,
    outdir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Path, Path, Path]:
    # Imports inside function (better for PyInstaller + faster GUI startup)
    from shield.demo.scenarios import make_demo_forcing, list_demo_scenarios
    from shield.core.state import BuildingProfile
    from shield.evaluation.runner import evaluate_policies

    if scenario not in list_demo_scenarios():
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list_demo_scenarios()}")

    # Keep deterministic “demo date” (matches your existing tooling)
    start_dt = datetime(2026, 1, 16, int(start_hour), 0, 0)
    forcing = make_demo_forcing(start=start_dt, horizon_hours=int(horizon_hours), scenario=scenario)

    building = BuildingProfile(
        archetype=str(archetype),
        floor_area_m2=float(area_m2),
        has_hepa=True,
        has_fan=True,
        occupants=int(occupants),
    )

    # Policies: robust + heuristic. Runner can include baselines automatically too.
    policies = ["robust", "heuristic"]
    if include_baselines:
        policies = ["robust", "heuristic", "always_closed", "always_open", "night_vent", "hepa_always"]

    rows, results = evaluate_policies(
        forcing,
        building,
        policies=policies,
        robust_kwargs=dict(robust_kwargs),
        include_baselines=bool(include_baselines),
        hepa_budget_h_per_day=hepa_budget_h_per_day,
    )

    stamp = _now_stamp()
    tag = f"{scenario}_{archetype}_A{int(round(area_m2))}_H{int(start_hour)}_M{int(occupants)}_{stamp}"

    csv_path = outdir / f"eval_{tag}.csv"
    json_path = outdir / f"eval_{tag}.json"
    html_path = outdir / f"report_{tag}.html"

    _write_csv(csv_path, rows)
    _write_json(json_path, {"rows": rows, "results": results, "robust_kwargs": robust_kwargs})

    meta = {
        "scenario": scenario,
        "archetype": archetype,
        "area_m2": area_m2,
        "start_hour": start_hour,
        "horizon_hours": horizon_hours,
        "occupants": occupants,
        "hepa_budget_h_per_day": "unlimited" if hepa_budget_h_per_day is None else hepa_budget_h_per_day,
        "include_baselines": include_baselines,
        "robust_kwargs": robust_kwargs,
        "written_csv": str(csv_path),
        "written_json": str(json_path),
    }
    _write_html_report(html_path, title=f"SHIELD Report — {scenario} / {archetype}", rows=rows, meta=meta)

    return rows, results, csv_path, json_path, html_path


# ----------------------------
# GUI
# ----------------------------

class ShieldGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SHIELD — Smoke & Heat Indoor Exposure Limiting Decision-tool")
        self.geometry("980x680")

        self._last_report: Optional[Path] = None
        self._last_outdir: Optional[Path] = None

        self._build_widgets()
        self._load_defaults()

    def _build_widgets(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        # Top: inputs
        inputs = ttk.LabelFrame(root, text="Inputs", padding=10)
        inputs.pack(fill=tk.X)

        def add_row(r: int, label: str, widget: tk.Widget) -> None:
            ttk.Label(inputs, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
            widget.grid(row=r, column=1, sticky="we", pady=4)

        inputs.columnconfigure(1, weight=1)

        # Scenario dropdown
        self.scenario_var = tk.StringVar()
        self.scenario_box = ttk.Combobox(inputs, textvariable=self.scenario_var, state="readonly")

        # Archetype (free text, but with suggestions)
        self.archetype_var = tk.StringVar()
        self.archetype_box = ttk.Combobox(inputs, textvariable=self.archetype_var)
        self.archetype_box["values"] = ("classroom", "home", "clinic")

        # Numeric fields
        self.area_var = tk.StringVar()
        self.start_var = tk.StringVar()
        self.horizon_var = tk.StringVar()
        self.occupants_var = tk.StringVar()
        self.hepa_budget_var = tk.StringVar()

        self.area_entry = ttk.Entry(inputs, textvariable=self.area_var)
        self.start_entry = ttk.Entry(inputs, textvariable=self.start_var)
        self.horizon_entry = ttk.Entry(inputs, textvariable=self.horizon_var)
        self.occupants_entry = ttk.Entry(inputs, textvariable=self.occupants_var)
        self.hepa_budget_entry = ttk.Entry(inputs, textvariable=self.hepa_budget_var)

        # Baselines checkbox
        self.baselines_var = tk.BooleanVar(value=True)
        self.baselines_chk = ttk.Checkbutton(inputs, text="Include baselines", variable=self.baselines_var)

        # Robust knobs
        knobs = ttk.LabelFrame(root, text="Robust planner knobs (optional)", padding=10)
        knobs.pack(fill=tk.X, pady=(10, 0))
        knobs.columnconfigure(1, weight=1)

        self.seed_var = tk.StringVar()
        self.lookahead_var = tk.StringVar()
        self.beam_var = tk.StringVar()
        self.rollout_var = tk.StringVar()
        self.risk_var = tk.StringVar()
        self.cvar_var = tk.StringVar()

        self.seed_entry = ttk.Entry(knobs, textvariable=self.seed_var)
        self.lookahead_entry = ttk.Entry(knobs, textvariable=self.lookahead_var)
        self.beam_entry = ttk.Entry(knobs, textvariable=self.beam_var)
        self.rollout_entry = ttk.Entry(knobs, textvariable=self.rollout_var)
        self.risk_entry = ttk.Entry(knobs, textvariable=self.risk_var)
        self.cvar_entry = ttk.Entry(knobs, textvariable=self.cvar_var)

        def add_knob(r: int, label: str, widget: tk.Widget) -> None:
            ttk.Label(knobs, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
            widget.grid(row=r, column=1, sticky="we", pady=4)

        # Output dir
        outframe = ttk.LabelFrame(root, text="Output", padding=10)
        outframe.pack(fill=tk.X, pady=(10, 0))
        outframe.columnconfigure(1, weight=1)
        self.outdir_var = tk.StringVar()
        self.outdir_entry = ttk.Entry(outframe, textvariable=self.outdir_var)

        # Buttons
        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, pady=(10, 0))

        self.run_btn = ttk.Button(btns, text="Run SHIELD", command=self._on_run)
        self.open_report_btn = ttk.Button(btns, text="Open last report", command=self._open_last_report, state=tk.DISABLED)
        self.open_folder_btn = ttk.Button(btns, text="Open output folder", command=self._open_outdir, state=tk.DISABLED)

        self.run_btn.pack(side=tk.LEFT)
        self.open_report_btn.pack(side=tk.LEFT, padx=8)
        self.open_folder_btn.pack(side=tk.LEFT)

        # Log / results
        results = ttk.LabelFrame(root, text="Results / Log", padding=10)
        results.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.text = tk.Text(results, height=16, wrap=tk.NONE)
        self.text.pack(fill=tk.BOTH, expand=True)

        # Place rows
        add_row(0, "Scenario", self.scenario_box)
        add_row(1, "Archetype", self.archetype_box)
        add_row(2, "Area (m²)", self.area_entry)
        add_row(3, "Start hour (0-23)", self.start_entry)
        add_row(4, "Horizon hours", self.horizon_entry)
        add_row(5, "Occupants", self.occupants_entry)
        add_row(6, "HEPA budget (h/day; 0=unlimited)", self.hepa_budget_entry)
        self.baselines_chk.grid(row=7, column=1, sticky="w", pady=(6, 0))

        add_knob(0, "Seed", self.seed_entry)
        add_knob(1, "Lookahead (hours)", self.lookahead_entry)
        add_knob(2, "Beam width", self.beam_entry)
        add_knob(3, "Rollout members", self.rollout_entry)
        add_knob(4, "Risk λ (0..1)", self.risk_entry)
        add_knob(5, "CVaR α (0..1)", self.cvar_entry)

        ttk.Label(outframe, text="Output folder").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.outdir_entry.grid(row=0, column=1, sticky="we", pady=4)

    def _load_defaults(self) -> None:
        # Scenario list (safe import inside)
        try:
            from shield.demo.scenarios import list_demo_scenarios
            scenarios = list_demo_scenarios()
        except Exception:
            scenarios = ["smoke_heat_day", "smoke_only", "heat_only", "mild"]

        self.scenario_box["values"] = scenarios
        self.scenario_var.set(scenarios[0] if scenarios else "smoke_heat_day")

        self.archetype_var.set("classroom")
        self.area_var.set("90")
        self.start_var.set("0")
        self.horizon_var.set("72")
        self.occupants_var.set("25")
        self.hepa_budget_var.set("2")

        # Match your common robust_kwargs defaults
        self.seed_var.set("42")
        self.lookahead_var.set("12")
        self.beam_var.set("10")
        self.rollout_var.set("25")
        self.risk_var.set("0.70")
        self.cvar_var.set("0.90")

        self.outdir_var.set("outputs")

    def _log(self, s: str) -> None:
        self.text.insert(tk.END, s + "\n")
        self.text.see(tk.END)

    def _parse_float(self, name: str, s: str) -> float:
        try:
            return float(s)
        except Exception:
            raise ValueError(f"{name} must be a number (got {s!r})")

    def _parse_int(self, name: str, s: str) -> int:
        try:
            return int(float(s))
        except Exception:
            raise ValueError(f"{name} must be an integer (got {s!r})")

    def _on_run(self) -> None:
        self.run_btn.config(state=tk.DISABLED)
        self.open_report_btn.config(state=tk.DISABLED)
        self.open_folder_btn.config(state=tk.DISABLED)
        self._last_report = None
        self._last_outdir = None

        self.text.delete("1.0", tk.END)

        # Collect inputs now (fast fail)
        try:
            scenario = self.scenario_var.get().strip()
            archetype = self.archetype_var.get().strip() or "classroom"
            area = self._parse_float("Area (m²)", self.area_var.get())
            start_hour = self._parse_int("Start hour", self.start_var.get())
            horizon = self._parse_int("Horizon hours", self.horizon_var.get())
            occupants = self._parse_int("Occupants", self.occupants_var.get())
            hepa_budget = self._parse_float("HEPA budget", self.hepa_budget_var.get())

            if not (0 <= start_hour <= 23):
                raise ValueError("Start hour must be between 0 and 23.")
            if horizon <= 0:
                raise ValueError("Horizon hours must be > 0.")
            if area <= 0:
                raise ValueError("Area must be > 0.")
            if occupants <= 0:
                raise ValueError("Occupants must be > 0.")

            hepa_budget_h_per_day: Optional[float] = float(hepa_budget)
            if hepa_budget_h_per_day <= 0.0:
                hepa_budget_h_per_day = None

            robust_kwargs: Dict[str, Any] = {
                "seed": self._parse_int("Seed", self.seed_var.get()),
                "lookahead_h": self._parse_int("Lookahead", self.lookahead_var.get()),
                "beam": self._parse_int("Beam width", self.beam_var.get()),
                "rollout_members": self._parse_int("Rollout members", self.rollout_var.get()),
                "risk": float(self._parse_float("Risk λ", self.risk_var.get())),
                "cvar": float(self._parse_float("CVaR α", self.cvar_var.get())),
            }

            include_baselines = bool(self.baselines_var.get())
            outdir = Path(self.outdir_var.get().strip() or "outputs")

        except Exception as e:
            self.run_btn.config(state=tk.NORMAL)
            messagebox.showerror("Invalid input", str(e))
            return

        self._log("Running SHIELD...")
        self._log(f"  scenario={scenario} archetype={archetype} area={area} start={start_hour} horizon={horizon}")
        self._log(f"  occupants={occupants} HEPA_budget={'unlimited' if hepa_budget_h_per_day is None else hepa_budget_h_per_day}")
        self._log(f"  robust_kwargs={robust_kwargs}")
        self._log("")

        def worker() -> None:
            try:
                rows, results, csv_path, json_path, html_path = run_one_eval(
                    scenario=scenario,
                    archetype=archetype,
                    area_m2=area,
                    start_hour=start_hour,
                    horizon_hours=horizon,
                    occupants=occupants,
                    hepa_budget_h_per_day=hepa_budget_h_per_day,
                    robust_kwargs=robust_kwargs,
                    include_baselines=include_baselines,
                    outdir=outdir,
                )
                table = _pretty_table(rows)
                self.after(0, lambda: self._on_done(table, outdir, html_path, csv_path, json_path))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._on_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, table: str, outdir: Path, html_path: Path, csv_path: Path, json_path: Path) -> None:
        self._log("DONE.\n")
        self._log(table.rstrip("\n"))
        self._log("")
        self._log(f"Wrote:\n  - {csv_path}\n  - {json_path}\n  - {html_path}")

        self._last_report = html_path
        self._last_outdir = outdir

        self.run_btn.config(state=tk.NORMAL)
        self.open_report_btn.config(state=tk.NORMAL)
        self.open_folder_btn.config(state=tk.NORMAL)

        # Auto-open report (feels “app-like”)
        try:
            webbrowser.open(html_path.resolve().as_uri())
        except Exception:
            pass

    def _on_error(self, err: str) -> None:
        self._log("ERROR:\n" + err)
        self.run_btn.config(state=tk.NORMAL)
        messagebox.showerror("SHIELD error", "Run failed. See log for details.")

    def _open_last_report(self) -> None:
        if not self._last_report:
            return
        p = self._last_report.resolve()
        try:
            webbrowser.open(p.as_uri())
        except Exception:
            try:
                os.startfile(str(p))  # type: ignore[attr-defined]
            except Exception as e:
                messagebox.showerror("Open report failed", str(e))

    def _open_outdir(self) -> None:
        if not self._last_outdir:
            return
        p = self._last_outdir.resolve()
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showerror("Open folder failed", str(e))


def main() -> int:
    app = ShieldGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
