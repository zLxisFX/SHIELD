"""
Robust Greedy Optimizer (UQ-aware action planner).

UPGRADE (IMPORTANT):
- Replaced "candidate first step + heuristic rollout" with a BEAM SEARCH lookahead.
  This makes robust genuinely different from heuristic and able to exploit short safe
  cooling windows (pre-cooling) without copying the heuristic behavior.

Core idea:
- Maintain an ensemble of uncertain PM + thermal parameters.
- At each hour, run a beam search over action sequences for L hours using a SUBSET
  of ensemble members for speed.
- Rank sequences by risk-aware objective:
    obj = (1-lam)*mean(cost) + lam*CVaR_alpha(cost)
- Take the first action from the best sequence, then advance the REAL full ensemble.

This gives a real planning separation while staying feasible for an offline demo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

from shield.core.state import ActionsHour, BuildingProfile, ForcingHour

# Physics
from shield.models.pm_mass_balance import PMModelParams, pm_step_with_controls
from shield.models.thermal_2r2c import ThermalParams, internal_gains_w, thermal_step_2r2c
from shield.models.indoor_sources import indoor_source_ug_h

# Priors for sampling
from shield.uq.priors import Priors, sample_joint_params


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
    # Harm weights (continuous exceedance penalties)
    w_pm: float = 1.0
    w_heat: float = 1.2

    # "Near-threshold" risk shaping (encourages pre-cooling / pre-cleaning)
    # These penalties activate before thresholds are crossed.
    w_pm_near: float = 0.20
    w_heat_near: float = 0.35
    pm_near_margin: float = 3.0      # start penalizing at (pm_thr - margin)
    heat_near_margin_c: float = 1.5  # start penalizing at (heat_thr - margin)

    # Small penalties to discourage unnecessary actions
    w_hepa: float = 0.02
    w_fan: float = 0.02
    w_windows: float = 0.01

    # Switching penalty (discourages toggling)
    w_switch: float = 0.01

    # Lookahead hours (beam search depth)
    lookahead_h: int = 8

    # Beam search width (kept beams per step)
    beam_width: int = 10

    # For speed: number of ensemble members used in lookahead search
    rollout_members: int = 25

    # Risk (robustness) control
    cvar_alpha: float = 0.90
    risk_lambda: float = 0.55  # 0 => mean only, 1 => CVaR only


# ---------------------------
# Helpers
# ---------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _candidate_actions(building: BuildingProfile) -> List[Tuple[bool, bool, bool]]:
    """
    Enumerate feasible (windows_open, hepa_on, fan_on).
    """
    windows_opts = [False, True]
    hepa_opts = [False, True] if building.has_hepa else [False]
    fan_opts = [False, True] if building.has_fan else [False]

    cands: List[Tuple[bool, bool, bool]] = []
    for w in windows_opts:
        for h in hepa_opts:
            for f in fan_opts:
                cands.append((w, h, f))
    return cands


def _stage_cost(
    pm_in: float,
    t_air: float,
    *,
    pm_threshold: float,
    heat_threshold_c: float,
    windows_open: bool,
    hepa_on: bool,
    fan_on: bool,
    prev_action: Tuple[bool, bool, bool] | None,
    weights: OptimizerWeights,
) -> float:
    """
    Continuous penalty for exceedance + near-threshold shaping + small action/switch cost.
    """
    pm_in = float(pm_in)
    t_air = float(t_air)
    pm_thr = float(pm_threshold)
    ht_thr = float(heat_threshold_c)

    # exceedance penalties
    pm_excess = max(0.0, pm_in - pm_thr)
    heat_excess = max(0.0, t_air - ht_thr)

    # near-threshold shaping (pre-emptive)
    pm_near_start = pm_thr - float(weights.pm_near_margin)
    heat_near_start = ht_thr - float(weights.heat_near_margin_c)

    pm_near = max(0.0, pm_in - pm_near_start)
    heat_near = max(0.0, t_air - heat_near_start)

    cost = 0.0
    cost += weights.w_pm * pm_excess
    cost += weights.w_heat * heat_excess

    cost += weights.w_pm_near * pm_near
    cost += weights.w_heat_near * heat_near

    # tiny "effort" penalties
    cost += weights.w_windows * (1.0 if windows_open else 0.0)
    cost += weights.w_hepa * (1.0 if hepa_on else 0.0)
    cost += weights.w_fan * (1.0 if fan_on else 0.0)

    # switching penalty
    if prev_action is not None:
        pw, ph, pf = prev_action
        flips = 0.0
        flips += 1.0 if bool(pw) != bool(windows_open) else 0.0
        flips += 1.0 if bool(ph) != bool(hepa_on) else 0.0
        flips += 1.0 if bool(pf) != bool(fan_on) else 0.0
        cost += weights.w_switch * flips

    return float(cost)


def _cvar(costs: List[float], alpha: float) -> float:
    """
    CVaR_alpha: average of the worst (1-alpha) fraction.
    alpha in (0,1). If alpha=0.90 => mean of worst 10%.
    """
    if not costs:
        return 0.0
    alpha = float(_clamp(alpha, 0.0, 0.999999))
    xs = sorted(float(c) for c in costs)
    n = len(xs)
    start = int(math.floor(alpha * n))
    start = min(max(0, start), n - 1)
    tail = xs[start:]
    return float(sum(tail) / len(tail))


def _risk_objective(costs: List[float], weights: OptimizerWeights) -> float:
    if not costs:
        return 0.0
    mean_cost = float(sum(costs) / max(1, len(costs)))
    tail_cost = _cvar(costs, alpha=weights.cvar_alpha)
    lam = float(_clamp(weights.risk_lambda, 0.0, 1.0))
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
    """
    One-hour physics step for a single member state.
    """
    # PM
    pm_next = pm_step_with_controls(
        params=pm_params,
        c_prev=pm_in,
        c_out=f.pm25_out,
        dt_h=1.0,
        windows_open=windows_open,
        hepa_on=hepa_on,
        indoor_source_ug_h=indoor_source_ug_h(building.archetype, building.occupants, f.t),
    )

    # Thermal (fan does not change air temperature in this simple model)
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


@dataclass
class _Beam:
    # per-member states (for rollout subset)
    pm_in: List[float]
    t_air: List[float]
    t_mass: List[float]
    # per-member accumulated costs
    costs: List[float]
    # first action in the sequence (the one we will execute)
    first_action: Tuple[bool, bool, bool] | None
    # last action (for switching penalty)
    last_action: Tuple[bool, bool, bool]
    # Track HEPA budget usage within this beam path
    hepa_used_by_day: Dict[int, float]


def _beam_search_best_first_action(
    members: List[EnsembleMember],
    forcing: List[ForcingHour],
    start_idx: int,
    *,
    building: BuildingProfile,
    pm_threshold: float,
    heat_threshold_c: float,
    weights: OptimizerWeights,
    prev_action: Tuple[bool, bool, bool] | None,
    hepa_budget_h_per_day: float | None,
    hepa_used_by_day_seed: Dict[int, float],
) -> Tuple[bool, bool, bool]:
    """
    Beam search over action sequences of length L, returning the best FIRST action.
    Uses only a subset of ensemble members for speed (weights.rollout_members).
    """
    if start_idx >= len(forcing):
        return (False, False, False)

    # Subset for rollout speed
    M = int(max(1, min(len(members), int(weights.rollout_members))))
    pm_params_list = [members[i].pm_params for i in range(M)]
    th_params_list = [members[i].th_params for i in range(M)]

    pm0 = [float(members[i].pm_in) for i in range(M)]
    ta0 = [float(members[i].t_air) for i in range(M)]
    tm0 = [float(members[i].t_mass) for i in range(M)]

    # Initialize beam
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
    L = int(max(1, weights.lookahead_h))
    end = min(len(forcing), start_idx + L)
    K = int(max(1, weights.beam_width))

    for k in range(start_idx, end):
        f = forcing[k]
        new_beams: List[_Beam] = []

        for b in beams:
            for (w_open, hepa_on, fan_on) in cands:
                # Check budget constraint within the lookahead
                if hepa_budget_h_per_day is not None and hepa_on:
                    day = k // 24
                    used = float(b.hepa_used_by_day.get(day, 0.0))
                    if used >= hepa_budget_h_per_day:
                        continue  # prune: candidate is invalid

                pm_next: List[float] = []
                ta_next: List[float] = []
                tm_next: List[float] = []
                c_next: List[float] = []

                for i in range(M):
                    pm_i, ta_i, tm_i = _step_member(
                        pm_params_list[i],
                        th_params_list[i],
                        b.pm_in[i],
                        b.t_air[i],
                        b.t_mass[i],
                        f,
                        building=building,
                        windows_open=w_open,
                        hepa_on=hepa_on,
                        fan_on=fan_on,
                    )

                    step_c = _stage_cost(
                        pm_i,
                        ta_i,
                        pm_threshold=pm_threshold,
                        heat_threshold_c=heat_threshold_c,
                        windows_open=w_open,
                        hepa_on=hepa_on,
                        fan_on=fan_on,
                        prev_action=b.last_action,
                        weights=weights,
                    )

                    pm_next.append(pm_i)
                    ta_next.append(ta_i)
                    tm_next.append(tm_i)
                    c_next.append(b.costs[i] + step_c)

                first = b.first_action
                if first is None:
                    first = (w_open, hepa_on, fan_on)

                # Update budget tracking for next step
                new_used = dict(b.hepa_used_by_day)
                if hepa_budget_h_per_day is not None and hepa_on:
                    day = k // 24
                    new_used[day] = float(new_used.get(day, 0.0)) + 1.0

                new_beams.append(
                    _Beam(
                        pm_in=pm_next,
                        t_air=ta_next,
                        t_mass=tm_next,
                        costs=c_next,
                        first_action=first,
                        last_action=(w_open, hepa_on, fan_on),
                        hepa_used_by_day=new_used,
                    )
                )

        # prune to top-K beams by risk objective
        new_beams.sort(key=lambda bb: _risk_objective(bb.costs, weights))
        beams = new_beams[:K]

        if not beams:
            break

    # Choose best beam
    best = min(beams, key=lambda bb: _risk_objective(bb.costs, weights))
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
) -> List[ActionsHour]:
    """
    Produce an hour-by-hour plan using robust planning with beam-search lookahead.

    Deterministic given the same:
    - forcing
    - building
    - seed
    - n_ensemble
    - priors
    - weights
    """
    if not forcing:
        raise ValueError("forcing is empty")
    if n_ensemble <= 0:
        raise ValueError("n_ensemble must be > 0")

    priors = priors or Priors.defaults()

    if weights is None:
        weights = OptimizerWeights()
    elif isinstance(weights, dict):
        weights = OptimizerWeights(**weights)

    # Build ensemble
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
    
    # Real-time budget tracking
    hepa_used_by_day: Dict[int, float] = {}
    budget = float(max(0.0, hepa_budget_h_per_day)) if hepa_budget_h_per_day is not None else None

    for idx, f in enumerate(forcing):
        # Choose action via beam search lookahead
        best_action = _beam_search_best_first_action(
            members,
            forcing,
            idx,
            building=building,
            pm_threshold=pm_threshold,
            heat_threshold_c=heat_threshold_c,
            weights=weights,
            prev_action=prev_action,
            hepa_budget_h_per_day=budget,
            hepa_used_by_day_seed=dict(hepa_used_by_day),
        )

        windows_open, hepa_on, fan_on = best_action
        
        # Update real budget usage
        day_now = idx // 24
        if budget is not None:
            if hepa_on:
                used = float(hepa_used_by_day.get(day_now, 0.0))
                # Safety check: if beam somehow returned invalid action
                if used >= budget:
                    hepa_on = False
                else:
                    hepa_used_by_day[day_now] = used + 1.0

        # Apply chosen action to ALL ensemble members (real state advance)
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