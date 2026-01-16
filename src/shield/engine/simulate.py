"""
Simulation engine (current version).

Capabilities:
- Indoor PM: physics single-zone mass balance (exact solution)
- Indoor heat: 2R2C thermal model (substepped Euler)
- Indoor sources: time-of-day dependent emissions
- Optional constraints:
    - HEPA runtime budget (hours/day) enforced consistently across policies
- Policy options:
    - "heuristic" (default): simple if/else rules
    - "robust_greedy": UQ-aware robust greedy planner (risk-based)

Still evolving:
- Uncertainty: ensemble bands are implemented separately; optimizer uses UQ priors
- True MPC / chance constraints: future upgrade
"""

from __future__ import annotations

import math
from dataclasses import is_dataclass, replace
from typing import List, Tuple, Literal, Optional, Dict, Any

from shield.core.state import (
    ActionsHour,
    BuildingProfile,
    ForcingHour,
    OutputHour,
    RunResult,
    validate_forcing,
)

# PM model (physics)
from shield.models.pm_mass_balance import make_params as make_pm_params
from shield.models.pm_mass_balance import pm_step_with_controls

# Thermal model (2R2C)
from shield.models.thermal_2r2c import make_params as make_th_params
from shield.models.thermal_2r2c import internal_gains_w, thermal_step_2r2c

# Indoor sources
from shield.models.indoor_sources import indoor_source_ug_h

# Robust optimizer (optional policy)
from shield.optimizer.robust_greedy import robust_greedy_plan

from shield.policies.heuristic import choose_actions_for_hour


WHO_PM25_24H_GUIDELINE_UG_M3 = 15.0  # WHO 2021 24h guideline (MVP threshold)
HEAT_THRESHOLD_C = 32.0             # simple heat threshold for MVP (will evolve)

PolicyName = Literal["heuristic", "robust_greedy"]


def compute_exceedance_metrics(
    outputs: List[OutputHour],
    pm25_threshold: float,
    heat_threshold_c: float,
) -> Tuple[int, int]:
    minutes_pm = 0
    hours_heat = 0
    for o in outputs:
        if o.pm25_in > pm25_threshold:
            minutes_pm += 60
        if o.temp_in_c >= heat_threshold_c:
            hours_heat += 1
    return minutes_pm, hours_heat


def _simulate_given_actions(
    forcing: List[ForcingHour],
    building: BuildingProfile,
    actions: List[ActionsHour],
    *,
    pm25_threshold: float,
    heat_threshold_c: float,
    pm25_init: float,
    temp_init_c: float,
) -> RunResult:
    """
    Run physics forward using a precomputed action schedule.
    
    NOTE: This uses the exact actions passed in. Any budget/constraint overrides
    must be applied to 'actions' BEFORE calling this function.
    """
    horizon_hours = len(forcing)
    if len(actions) != horizon_hours:
        raise ValueError(f"actions length {len(actions)} must match forcing length {horizon_hours}")

    validate_forcing(forcing, horizon_hours)

    pm_params = make_pm_params(building.archetype, building.floor_area_m2)
    th_params = make_th_params(building.archetype, building.floor_area_m2)

    pm25_in = float(max(0.0, pm25_init))
    t_air = float(temp_init_c)
    t_mass = float(temp_init_c)

    outputs: List[OutputHour] = []
    warnings: List[str] = []

    for f, a in zip(forcing, actions):
        # Apply physics using the action as-is
        pm25_in = pm_step_with_controls(
            params=pm_params,
            c_prev=pm25_in,
            c_out=f.pm25_out,
            dt_h=1.0,
            windows_open=a.windows_open,
            hepa_on=a.hepa_on,
            indoor_source_ug_h=indoor_source_ug_h(building.archetype, building.occupants, f.t),
        )

        q_int_w = internal_gains_w(building.archetype, building.occupants, f.t)
        t_air, t_mass = thermal_step_2r2c(
            params=th_params,
            t_air_prev_c=t_air,
            t_mass_prev_c=t_mass,
            t_out_c=f.temp_out_c,
            dt_h=1.0,
            windows_open=a.windows_open,
            q_internal_w=q_int_w,
            substeps=12,
        )

        outputs.append(OutputHour(t=f.t, pm25_in=pm25_in, temp_in_c=t_air))

    minutes_pm, hours_heat = compute_exceedance_metrics(outputs, pm25_threshold, heat_threshold_c)

    if not building.has_hepa:
        warnings.append("No HEPA/filter selected; smoke risk may be higher.")
    if building.occupants >= 30:
        warnings.append("High occupancy increases heat/CO2; CO2 not modeled yet.")

    return RunResult(
        start=forcing[0].t,
        horizon_hours=horizon_hours,
        building=building,
        forcing=forcing,
        actions=actions,
        outputs=outputs,
        minutes_pm25_above_threshold=minutes_pm,
        hours_heat_above_threshold=hours_heat,
        pm25_threshold=pm25_threshold,
        heat_threshold_c=heat_threshold_c,
        warnings=warnings,
    )


def _apply_smoke_guard_to_actions(
    forcing: List[ForcingHour],
    actions: List[Any],
    *,
    pm_threshold: float,
    safety_factor: float = 0.70,
) -> List[Any]:
    """
    Safety guard: if an action opens windows while outdoor PM is too high,
    override to windows CLOSED and add a note.

    This prevents the robust policy from doing "slightly worse" than heuristic
    on smoke when heuristic already attains 0 exceedance minutes.
    """
    pm_open_max = float(pm_threshold) * float(safety_factor)

    fixed: List[Any] = []
    n = min(len(forcing), len(actions))

    for i in range(len(actions)):
        a = actions[i]
        if i >= n:
            fixed.append(a)
            continue

        f = forcing[i]
        windows_open = bool(getattr(a, "windows_open", False))
        if not windows_open:
            fixed.append(a)
            continue

        if float(f.pm25_out) <= pm_open_max:
            fixed.append(a)
            continue

        note = f"Override: windows CLOSED (outdoor PM {f.pm25_out:.1f} > {pm_open_max:.1f})"
        notes = list(getattr(a, "notes", []) or [])

        # If we override to CLOSED, remove any earlier "open" notes to avoid contradictions
        notes = [n for n in (notes or []) if "windows OPEN" not in n and "Ventilate (windows OPEN)" not in n]

        notes.append(note)

        # Dataclass-safe override (works if Action is frozen or mutable)
        if is_dataclass(a):
            fields = getattr(a, "__dataclass_fields__", {})
            kwargs = {}
            if "windows_open" in fields:
                kwargs["windows_open"] = False
            if "notes" in fields:
                kwargs["notes"] = notes
            try:
                a2 = replace(a, **kwargs)
                fixed.append(a2)
                continue
            except Exception:
                pass  # fall through to mutation attempt

        # Mutable object fallback
        try:
            setattr(a, "windows_open", False)
            if hasattr(a, "notes"):
                setattr(a, "notes", notes)
        except Exception:
            pass

        fixed.append(a)

    return fixed


def _apply_hepa_budget_to_actions(
    actions: List[Any],
    *,
    hepa_budget_h_per_day: float | None,
) -> List[Any]:
    """
    Enforce HEPA runtime budget by MODIFYING the actions list (so exports/compare_plans are truthful).

    Rule: each 24-hour block gets `hepa_budget_h_per_day` hours of HEPA.
    If budget is exhausted, set hepa_on=False and add an override note.
    """
    if hepa_budget_h_per_day is None:
        return actions
    budget = float(hepa_budget_h_per_day)
    if budget < 0:
        return actions

    fixed: List[Any] = []
    remaining = budget

    for i, a in enumerate(actions):
        # reset budget every "day" (each 24-hour chunk)
        if i % 24 == 0:
            remaining = budget

        hepa_on = bool(getattr(a, "hepa_on", False))
        if not hepa_on:
            fixed.append(a)
            continue

        # allow if budget remains
        if remaining >= 1.0:
            remaining -= 1.0
            fixed.append(a)
            continue

        # otherwise override OFF
        notes = list(getattr(a, "notes", []) or [])
        
        # Make notes consistent
        notes = [n for n in (notes or []) if n.strip() != "Run HEPA / filter"]
        notes.append("Override: HEPA OFF (budget exhausted)")

        # IMPORTANT: record the actual executed action (HEPA OFF), not the requested one
        if is_dataclass(a):
            fields = getattr(a, "__dataclass_fields__", {})
            kwargs = {}
            if "hepa_on" in fields:
                kwargs["hepa_on"] = False
            if "notes" in fields:
                kwargs["notes"] = notes
            try:
                a = replace(a, **kwargs)
            except Exception:
                pass
        else:
            try:
                setattr(a, "hepa_on", False)
                if hasattr(a, "notes"):
                    setattr(a, "notes", notes)
            except Exception:
                pass

        fixed.append(a)

    return fixed


def simulate_demo(
    forcing: List[ForcingHour],
    building: BuildingProfile,
    pm25_threshold: float = WHO_PM25_24H_GUIDELINE_UG_M3,
    heat_threshold_c: float = HEAT_THRESHOLD_C,
    pm25_init: float = 8.0,
    temp_init_c: float = 28.5,
    *,
    policy: PolicyName = "heuristic",
    policy_kwargs: Optional[Dict[str, Any]] = None,
    hepa_budget_h_per_day: float | None = None,
) -> RunResult:
    """
    Run a simulation over the forcing horizon.

    policy:
      - "heuristic": choose actions step-by-step using the built-in heuristic
      - "robust_greedy": compute full schedule using UQ-aware optimizer, then simulate

    policy_kwargs: optional dict passed to the optimizer (for robust_greedy).
    hepa_budget_h_per_day: optional HEPA runtime constraint (hours/day).
    """
    horizon_hours = len(forcing)
    if horizon_hours == 0:
        raise ValueError("Forcing is empty")

    validate_forcing(forcing, horizon_hours)

    # --- Policy selection ---
    if policy == "robust_greedy":
        kw = dict(policy_kwargs or {})
        actions = robust_greedy_plan(
            forcing=forcing,
            building=building,
            pm_threshold=pm25_threshold,
            heat_threshold_c=heat_threshold_c,
            pm25_init=pm25_init,
            temp_init_c=temp_init_c,
            hepa_budget_h_per_day=hepa_budget_h_per_day,
            **kw,
        )

        # Smoke guard: never open windows when outdoor PM is above a safe bound
        actions = _apply_smoke_guard_to_actions(
            forcing,
            actions,
            pm_threshold=pm25_threshold,
            safety_factor=0.70,
        )

        # Enforce budget AFTER smoke guard
        actions = _apply_hepa_budget_to_actions(
            actions,
            hepa_budget_h_per_day=hepa_budget_h_per_day,
        )

        return _simulate_given_actions(
            forcing=forcing,
            building=building,
            actions=actions,
            pm25_threshold=pm25_threshold,
            heat_threshold_c=heat_threshold_c,
            pm25_init=pm25_init,
            temp_init_c=temp_init_c,
        )

    # Default: heuristic step-by-step (existing behavior)
    pm_params = make_pm_params(building.archetype, building.floor_area_m2)
    th_params = make_th_params(building.archetype, building.floor_area_m2)

    pm25_in = float(max(0.0, pm25_init))
    t_air = float(temp_init_c)
    t_mass = float(temp_init_c)

    actions: List[ActionsHour] = []
    outputs: List[OutputHour] = []
    warnings: List[str] = []

    for f in forcing:
        windows_open, hepa_on, fan_on, notes = choose_actions_for_hour(
            f=f,
            pm25_in_prev=pm25_in,
            temp_in_prev=t_air,
            building=building,
        )

        # HEPA daily budget enforcement (live, affects physics)
        if hepa_budget_h_per_day is not None:
            budget = float(max(0.0, hepa_budget_h_per_day))
            hour_idx = len(actions)  # current hour index
            day = hour_idx // 24
            if "_hepa_used_by_day" not in locals():
                _hepa_used_by_day = {}
            used = float(_hepa_used_by_day.get(day, 0.0))

            if hepa_on and used >= budget:
                # Budget override: force HEPA OFF and keep notes consistent
                hepa_on = False
                notes = [n for n in (notes or []) if n.strip() != "Run HEPA / filter"]
                notes.append("Override: HEPA OFF (budget exhausted)")
            elif hepa_on:
                _hepa_used_by_day[day] = used + 1.0

        pm25_in = pm_step_with_controls(
            params=pm_params,
            c_prev=pm25_in,
            c_out=f.pm25_out,
            dt_h=1.0,
            windows_open=windows_open,
            hepa_on=hepa_on,
            indoor_source_ug_h=indoor_source_ug_h(building.archetype, building.occupants, f.t),
        )

        q_int_w = internal_gains_w(building.archetype, building.occupants, f.t)
        t_air, t_mass = thermal_step_2r2c(
            params=th_params,
            t_air_prev_c=t_air,
            t_mass_prev_c=t_mass,
            t_out_c=f.temp_out_c,
            dt_h=1.0,
            windows_open=windows_open,
            q_internal_w=q_int_w,
            substeps=12,
        )

        actions.append(
            ActionsHour(
                t=f.t,
                windows_open=windows_open,
                hepa_on=hepa_on,
                fan_on=fan_on,
                notes=notes,
            )
        )

        outputs.append(OutputHour(t=f.t, pm25_in=pm25_in, temp_in_c=t_air))

    minutes_pm, hours_heat = compute_exceedance_metrics(outputs, pm25_threshold, heat_threshold_c)

    if not building.has_hepa:
        warnings.append("No HEPA/filter selected; smoke risk may be higher.")
    if building.occupants >= 30:
        warnings.append("High occupancy increases heat/CO2; CO2 not modeled yet.")
    if hepa_budget_h_per_day is not None:
        warnings.append(f"HEPA budget active: {hepa_budget_h_per_day} h/day.")

    return RunResult(
        start=forcing[0].t,
        horizon_hours=horizon_hours,
        building=building,
        forcing=forcing,
        actions=actions,
        outputs=outputs,
        minutes_pm25_above_threshold=minutes_pm,
        hours_heat_above_threshold=hours_heat,
        pm25_threshold=pm25_threshold,
        heat_threshold_c=heat_threshold_c,
        warnings=warnings,
    )