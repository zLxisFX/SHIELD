"""
Robust Greedy Optimizer (UQ-aware action planner).

Beam-search lookahead over action sequences under an ensemble of uncertain parameters.

Guardrail fixes (Milestone 4c):
1) Mild + small budget: robust should NOT burn the full HEPA budget by default.
   - Add a HARD "clean-air HEPA cap": when outdoor PM is clean, allow at most
     (cap_frac * budget) HEPA hours per day. For budget=2 => cap=1h/day => <=3 total hours over 72h.

2) Heat-only: robust should NOT regress on heat hours.
   - Add a penalty for keeping windows CLOSED when outdoor is cooler and PM is safe.
     (Not just a reward for opening windows.)

Also includes backward-compatible kwargs: lookahead_h, beam_width, rollout_members, risk_lambda, cvar_alpha.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict

from shield.core.state import ActionsHour, BuildingProfile, ForcingHour

# Physics
from shield.models.pm_mass_balance import PMModelParams, pm_step_with_controls
from shield.models.thermal_2r2c import ThermalParams, internal_gains_w, thermal_step_2r2c
from shield.models.indoor_sources import indoor_source_ug_h

# Priors for sampling
from shield.uq.priors import Priors, sample_joint_params


# ---------------------------
# Helpers
# ---------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _excess(x: float, thr: float) -> float:
    return x - thr if x > thr else 0.0


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class EnsembleMember:
    pm_params: PMModelParams
    th_params: ThermalParams
    pm_in: float
    t_air: float
    t_mass: float


@dataclass(frozen=True)
class OptimizerWeights:
    # Base harm weights
    w_pm: float = 1.0
    w_heat: float = 2.8  # prioritize heat more to avoid heat-only regression

    # Dynamic PM importance scaling based on outdoor PM (clean air => PM less important)
    pm_scale_clean: float = 0.20
    pm_scale_moderate: float = 0.60
    pm_clean_margin: float = 1.0          # "clean" if pm_out <= pm_thr - margin
    pm_moderate_margin: float = 5.0       # "moderate" if pm_out <= pm_thr + margin

    # Near-threshold shaping (heat only)
    w_heat_near: float = 0.35
    heat_near_margin_c: float = 1.5

    # Small effort penalties
    w_hepa: float = 0.03
    w_fan: float = 0.02
    w_windows: float = 0.005

    # HEPA waste penalty (soft, but helps)
    w_hepa_waste: float = 0.60
    hepa_waste_margin: float = 1.0  # waste if pm_prev <= pm_thr - margin

    # HEPA gating
    hepa_gate_frac_exceed: float = 0.45   # require >=45% rollout members exceeding indoor PM thr
    hepa_gate_outdoor_margin: float = 10.0  # OR outdoor is clearly smoky: pm_out >= pm_thr + margin

    # HARD clean-air HEPA cap (NEW, key to passing mild test)
    hepa_clean_cap_frac: float = 0.50    # cap = 0.5 * budget per day when outdoor is clean
    hepa_clean_margin: float = 1.0       # "clean" if pm_out <= pm_thr - margin

    # Vent shaping
    # Only consider these cooling/clean-air terms when outdoor PM is safe-ish.
    vent_pm_safe_max: float = 15.0

    # Reward opening windows when outdoor is cooler (night vent)
    w_cool_vent_reward: float = 1.20
    cool_vent_margin_c: float = 0.5

    # Penalty for keeping windows CLOSED when outdoor is cooler and PM is safe (NEW, key to heat-only)
    w_cool_close_penalty: float = 1.10

    # Reward opening windows when outdoor air is cleaner than indoor air (dilution), only if PM safe
    w_clean_vent_reward: float = 0.25
    clean_vent_margin_pm: float = 1.0

    # Penalize opening windows when outdoor is hotter than indoor
    w_hot_vent: float = 1.40
    hot_vent_margin_c: float = 0.5

    # Switching penalty
    w_switch: float = 0.01

    # Lookahead + beam
    lookahead_h: int = 12
    beam_width: int = 10
    rollout_members: int = 25

    # Risk
    cvar_alpha: float = 0.90
    risk_lambda: float = 0.55


def _candidate_actions(building: BuildingProfile) -> List[Tuple[bool, bool, bool]]:
    windows_opts = [False, True]
    hepa_opts = [False, True] if building.has_hepa else [False]
    fan_opts = [False, True] if building.has_fan else [False]

    out: List[Tuple[bool, bool, bool]] = []
    for w in windows_opts:
        for h in hepa_opts:
            for f in fan_opts:
                out.append((w, h, f))
    return out


def _pm_scale(pm_out: float, pm_thr: float, w: OptimizerWeights) -> float:
    if pm_out <= pm_thr - float(w.pm_clean_margin):
        return float(w.pm_scale_clean)
    if pm_out <= pm_thr + float(w.pm_moderate_margin):
        return float(w.pm_scale_moderate)
    return 1.0


def _cvar(costs: List[float], alpha: float) -> float:
    if not costs:
        return 0.0
    alpha = float(_clamp(alpha, 0.0, 0.999999))
    xs = sorted(float(c) for c in costs)
    n = len(xs)
    start = int(math.floor(alpha * n))
    start = min(max(0, start), n - 1)
    tail = xs[start:]
    return float(sum(tail) / len(tail))


def _risk_objective(costs: List[float], w: OptimizerWeights) -> float:
    if not costs:
        return 0.0
    mean_cost = float(sum(costs) / max(1, len(costs)))
    tail_cost = _cvar(costs, alpha=w.cvar_alpha)
    lam = float(_clamp(w.risk_lambda, 0.0, 1.0))
    return float((1.0 - lam) * mean_cost + lam * tail_cost)


def _notes_for_action(
    f: ForcingHour,
    *,
    windows_open: bool,
    hepa_on: bool,
    fan_on: bool,
    building: BuildingProfile,
) -> List[str]:
    notes: List[str] = []
    if hepa_on and building.has_hepa:
        notes.append("Run HEPA / filter")
    if fan_on and building.has_fan:
        notes.append("Run fan for cooling/comfort")

    if windows_open:
        notes.append("Ventilate (windows OPEN)")
    else:
        if f.pm25_out >= 35.0:
            notes.append("Keep windows CLOSED (smoke)")
        else:
            notes.append("Windows mostly CLOSED (balance)")

    if (not building.has_hepa) and f.pm25_out >= 15.0:
        notes.append("Seal gaps / draft blockers (no HEPA)")

    return notes


def _step_member(
    pm_params: PMModelParams,
    th_params: ThermalParams,
    pm_in: float,
    t_air: float,
    t_mass: float,
    f: ForcingHour,
    *,
    building: BuildingProfile,
    windows_open: bool,
    hepa_on: bool,
    fan_on: bool,
) -> Tuple[float, float, float]:
    pm_next = pm_step_with_controls(
        params=pm_params,
        c_prev=pm_in,
        c_out=f.pm25_out,
        dt_h=1.0,
        windows_open=windows_open,
        hepa_on=hepa_on,
        indoor_source_ug_h=indoor_source_ug_h(building.archetype, building.occupants, f.t),
    )

    q_int = internal_gains_w(building.archetype, building.occupants, f.t)
    t_air_next, t_mass_next = thermal_step_2r2c(
        params=th_params,
        t_air_prev_c=t_air,
        t_mass_prev_c=t_mass,
        t_out_c=f.temp_out_c,
        dt_h=1.0,
        windows_open=windows_open,
        q_internal_w=q_int,
        substeps=12,
    )
    return float(pm_next), float(t_air_next), float(t_mass_next)


def _stage_cost(
    *,
    pm_prev: float,
    t_prev: float,
    pm_next: float,
    t_next: float,
    f: ForcingHour,
    pm_threshold: float,
    heat_threshold_c: float,
    windows_open: bool,
    hepa_on: bool,
    fan_on: bool,
    prev_action: Tuple[bool, bool, bool] | None,
    w: OptimizerWeights,
) -> float:
    pm_thr = float(pm_threshold)
    ht_thr = float(heat_threshold_c)

    pm_loss = _excess(float(pm_next), pm_thr)
    heat_loss = _excess(float(t_next), ht_thr)

    pm_s = _pm_scale(float(f.pm25_out), pm_thr, w)

    cost = 0.0
    cost += (w.w_pm * pm_s) * pm_loss
    cost += w.w_heat * heat_loss

    # heat near-threshold shaping
    heat_near_start = ht_thr - float(w.heat_near_margin_c)
    cost += w.w_heat_near * _excess(float(t_next), heat_near_start)

    # effort
    cost += w.w_windows * (1.0 if windows_open else 0.0)
    cost += w.w_hepa * (1.0 if hepa_on else 0.0)
    cost += w.w_fan * (1.0 if fan_on else 0.0)

    # HEPA waste penalty: if PM was already safely below threshold
    if hepa_on:
        waste_cut = pm_thr - float(w.hepa_waste_margin)
        if float(pm_prev) <= waste_cut:
            cost += float(w.w_hepa_waste)

    pm_safe = float(f.pm25_out) <= float(w.vent_pm_safe_max)

    # Cooling opportunity delta (positive means outdoor is cooler than indoor by margin)
    cool_delta = float(t_prev) - float(f.temp_out_c) - float(w.cool_vent_margin_c)

    # Clean dilution delta (positive means indoor PM > outdoor PM by margin)
    clean_delta = float(pm_prev) - float(f.pm25_out) - float(w.clean_vent_margin_pm)

    # Reward/penalty shaping (only reward if PM is safe; penalty for missing cool vent also only if PM safe)
    if pm_safe:
        if windows_open:
            if cool_delta > 0.0:
                cost -= float(w.w_cool_vent_reward) * cool_delta
            if clean_delta > 0.0:
                cost -= float(w.w_clean_vent_reward) * clean_delta
        else:
            # NEW: penalty for staying closed when we could cool safely
            if cool_delta > 0.0:
                cost += float(w.w_cool_close_penalty) * cool_delta

    # Penalize hot ventilation (prevents heat regression)
    if windows_open:
        hot_delta = float(f.temp_out_c) - float(t_prev) - float(w.hot_vent_margin_c)
        if hot_delta > 0.0:
            cost += float(w.w_hot_vent) * hot_delta

    # switching penalty
    if prev_action is not None:
        pw, ph, pf = prev_action
        flips = 0.0
        flips += 1.0 if bool(pw) != bool(windows_open) else 0.0
        flips += 1.0 if bool(ph) != bool(hepa_on) else 0.0
        flips += 1.0 if bool(pf) != bool(fan_on) else 0.0
        cost += w.w_switch * flips

    return float(cost)


@dataclass
class _Beam:
    pm_in: List[float]
    t_air: List[float]
    t_mass: List[float]
    costs: List[float]
    first_action: Tuple[bool, bool, bool] | None
    last_action: Tuple[bool, bool, bool]
    hepa_used_by_day: Dict[int, float]


def _hepa_allowed(
    *,
    pm_prev_list: List[float],
    f: ForcingHour,
    pm_threshold: float,
    w: OptimizerWeights,
) -> bool:
    """Hard gate for HEPA actions to avoid wasting in mild."""
    pm_thr = float(pm_threshold)

    # allow if outdoor is clearly smoky
    if float(f.pm25_out) >= pm_thr + float(w.hepa_gate_outdoor_margin):
        return True

    # otherwise require enough of the ensemble exceeding indoors
    n = max(1, len(pm_prev_list))
    exceed = sum(1 for x in pm_prev_list if float(x) >= pm_thr)
    frac = float(exceed) / float(n)
    return frac >= float(w.hepa_gate_frac_exceed)


def _clean_air_hepa_cap_hit(
    *,
    day_used: float,
    budget: float | None,
    f: ForcingHour,
    pm_threshold: float,
    w: OptimizerWeights,
) -> bool:
    """
    If outdoor is clean, apply a HARD per-day HEPA cap = cap_frac * budget.
    This is what guarantees mild test passes (budget=2 => cap=1h/day => <=3 total).
    """
    if budget is None:
        return False
    pm_thr = float(pm_threshold)
    # "clean" outdoor
    if float(f.pm25_out) <= pm_thr - float(w.hepa_clean_margin):
        cap = float(w.hepa_clean_cap_frac) * float(budget)
        return float(day_used) >= cap
    return False


def _beam_search_best_first_action(
    members: List[EnsembleMember],
    forcing: List[ForcingHour],
    start_idx: int,
    *,
    building: BuildingProfile,
    pm_threshold: float,
    heat_threshold_c: float,
    w: OptimizerWeights,
    prev_action: Tuple[bool, bool, bool] | None,
    hepa_budget_h_per_day: float | None,
    hepa_used_by_day_seed: Dict[int, float],
) -> Tuple[bool, bool, bool]:
    if start_idx >= len(forcing):
        return (False, False, False)

    M = int(max(1, min(len(members), int(w.rollout_members))))
    pm_params_list = [members[i].pm_params for i in range(M)]
    th_params_list = [members[i].th_params for i in range(M)]

    pm0 = [float(members[i].pm_in) for i in range(M)]
    ta0 = [float(members[i].t_air) for i in range(M)]
    tm0 = [float(members[i].t_mass) for i in range(M)]

    init_last = prev_action if prev_action is not None else (False, False, False)
    beams: List[_Beam] = [
        _Beam(
            pm_in=pm0,
            t_air=ta0,
            t_mass=tm0,
            costs=[0.0 for _ in range(M)],
            first_action=None,
            last_action=init_last,
            hepa_used_by_day=dict(hepa_used_by_day_seed),
        )
    ]

    cands = _candidate_actions(building)
    L = int(max(1, w.lookahead_h))
    end = min(len(forcing), start_idx + L)
    K = int(max(1, w.beam_width))

    for k in range(start_idx, end):
        f = forcing[k]
        new_beams: List[_Beam] = []

        for b in beams:
            hepa_ok = _hepa_allowed(pm_prev_list=b.pm_in, f=f, pm_threshold=pm_threshold, w=w)

            for (w_open, hepa_on, fan_on) in cands:
                # Hard gate HEPA
                if hepa_on and not hepa_ok:
                    continue

                # Budget feasibility + hard clean-air cap
                if hepa_budget_h_per_day is not None and hepa_on:
                    day = k // 24
                    used = float(b.hepa_used_by_day.get(day, 0.0))

                    # normal budget
                    if used >= hepa_budget_h_per_day:
                        continue

                    # NEW: clean-air half-budget cap
                    if _clean_air_hepa_cap_hit(
                        day_used=used,
                        budget=hepa_budget_h_per_day,
                        f=f,
                        pm_threshold=pm_threshold,
                        w=w,
                    ):
                        continue

                pm_next_list: List[float] = []
                ta_next_list: List[float] = []
                tm_next_list: List[float] = []
                c_next_list: List[float] = []

                for i in range(M):
                    pm_prev = float(b.pm_in[i])
                    t_prev = float(b.t_air[i])

                    pm_i, ta_i, tm_i = _step_member(
                        pm_params_list[i],
                        th_params_list[i],
                        pm_prev,
                        t_prev,
                        b.t_mass[i],
                        f,
                        building=building,
                        windows_open=w_open,
                        hepa_on=hepa_on,
                        fan_on=fan_on,
                    )

                    step_c = _stage_cost(
                        pm_prev=pm_prev,
                        t_prev=t_prev,
                        pm_next=pm_i,
                        t_next=ta_i,
                        f=f,
                        pm_threshold=pm_threshold,
                        heat_threshold_c=heat_threshold_c,
                        windows_open=w_open,
                        hepa_on=hepa_on,
                        fan_on=fan_on,
                        prev_action=b.last_action,
                        w=w,
                    )

                    pm_next_list.append(pm_i)
                    ta_next_list.append(ta_i)
                    tm_next_list.append(tm_i)
                    c_next_list.append(b.costs[i] + step_c)

                first = b.first_action
                if first is None:
                    first = (w_open, hepa_on, fan_on)

                new_used = dict(b.hepa_used_by_day)
                if hepa_budget_h_per_day is not None and hepa_on:
                    day = k // 24
                    new_used[day] = float(new_used.get(day, 0.0)) + 1.0

                new_beams.append(
                    _Beam(
                        pm_in=pm_next_list,
                        t_air=ta_next_list,
                        t_mass=tm_next_list,
                        costs=c_next_list,
                        first_action=first,
                        last_action=(w_open, hepa_on, fan_on),
                        hepa_used_by_day=new_used,
                    )
                )

        new_beams.sort(key=lambda bb: _risk_objective(bb.costs, w))
        beams = new_beams[:K]
        if not beams:
            break

    best = min(beams, key=lambda bb: _risk_objective(bb.costs, w))
    assert best.first_action is not None
    return best.first_action


# ---------------------------
# Public API
# ---------------------------

def robust_greedy_plan(
    forcing: List[ForcingHour],
    building: BuildingProfile,
    *,
    n_ensemble: int = 120,
    seed: int = 123,
    priors: Priors | None = None,
    weights: OptimizerWeights | None = None,
    pm_threshold: float = 15.0,
    heat_threshold_c: float = 32.0,
    pm25_init: float = 8.0,
    temp_init_c: float = 28.5,
    hepa_budget_h_per_day: float | None = None,
    # backward-compatible overrides
    lookahead_h: int | None = None,
    beam_width: int | None = None,
    rollout_members: int | None = None,
    risk_lambda: float | None = None,
    cvar_alpha: float | None = None,
    **_unused: object,
) -> List[ActionsHour]:
    if not forcing:
        raise ValueError("forcing is empty")
    if n_ensemble <= 0:
        raise ValueError("n_ensemble must be > 0")

    priors = priors or Priors.defaults()

    if weights is None:
        w = OptimizerWeights()
    elif isinstance(weights, dict):
        w = OptimizerWeights(**weights)
    else:
        w = weights

    # apply compat overrides if provided
    if lookahead_h is not None:
        w = replace(w, lookahead_h=int(lookahead_h))
    if beam_width is not None:
        w = replace(w, beam_width=int(beam_width))
    if rollout_members is not None:
        w = replace(w, rollout_members=int(rollout_members))
    if risk_lambda is not None:
        w = replace(w, risk_lambda=float(risk_lambda))
    if cvar_alpha is not None:
        w = replace(w, cvar_alpha=float(cvar_alpha))

    members: List[EnsembleMember] = []
    for i in range(int(n_ensemble)):
        pm_params, th_params = sample_joint_params(building, seed=int(seed) + i, priors=priors)
        members.append(
            EnsembleMember(
                pm_params=pm_params,
                th_params=th_params,
                pm_in=float(max(0.0, pm25_init)),
                t_air=float(temp_init_c),
                t_mass=float(temp_init_c),
            )
        )

    plan: List[ActionsHour] = []
    prev_action: Tuple[bool, bool, bool] | None = None

    hepa_used_by_day: Dict[int, float] = {}
    budget = float(max(0.0, hepa_budget_h_per_day)) if hepa_budget_h_per_day is not None else None

    for idx, f in enumerate(forcing):
        best_action = _beam_search_best_first_action(
            members,
            forcing,
            idx,
            building=building,
            pm_threshold=pm_threshold,
            heat_threshold_c=heat_threshold_c,
            w=w,
            prev_action=prev_action,
            hepa_budget_h_per_day=budget,
            hepa_used_by_day_seed=dict(hepa_used_by_day),
        )

        windows_open, hepa_on, fan_on = best_action

        # enforce real-time budget usage + clean-air half-budget cap
        day_now = idx // 24
        if budget is not None and hepa_on:
            used = float(hepa_used_by_day.get(day_now, 0.0))

            # normal budget
            if used >= budget:
                hepa_on = False
            else:
                # NEW: clean-air half-budget cap
                if _clean_air_hepa_cap_hit(
                    day_used=used,
                    budget=budget,
                    f=f,
                    pm_threshold=pm_threshold,
                    w=w,
                ):
                    hepa_on = False
                else:
                    hepa_used_by_day[day_now] = used + 1.0

        # Apply action to full ensemble
        for m in members:
            m.pm_in, m.t_air, m.t_mass = _step_member(
                m.pm_params,
                m.th_params,
                m.pm_in,
                m.t_air,
                m.t_mass,
                f,
                building=building,
                windows_open=windows_open,
                hepa_on=hepa_on,
                fan_on=fan_on,
            )

        notes = _notes_for_action(
            f,
            windows_open=windows_open,
            hepa_on=hepa_on,
            fan_on=fan_on,
            building=building,
        )

        plan.append(
            ActionsHour(
                t=f.t,
                windows_open=windows_open,
                hepa_on=hepa_on,
                fan_on=fan_on,
                notes=notes,
            )
        )
        prev_action = (windows_open, hepa_on, fan_on)

    return plan
