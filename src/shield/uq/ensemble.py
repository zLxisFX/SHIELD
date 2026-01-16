"""
Ensemble simulation for SHIELD (uncertainty bands).

This module:
- Samples N plausible parameter sets from priors (seeded for reproducibility)
- Runs the simulator for each parameter set
- Produces per-hour quantiles for indoor PM and indoor temperature

This is offline-friendly (no internet required).
Later, "priors" become "posteriors" after Bayesian calibration with sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

from shield.core.state import ActionsHour, BuildingProfile, ForcingHour, OutputHour
from shield.engine import simulate as sim_engine
from shield.models.pm_mass_balance import PMModelParams
from shield.models.thermal_2r2c import ThermalParams, internal_gains_w, thermal_step_2r2c
from shield.models.pm_mass_balance import pm_step_with_controls
from shield.uq.priors import Priors, sample_joint_params


@dataclass(frozen=True)
class BandsHour:
    t: datetime

    pm50: float
    pm10: float
    pm90: float
    pm02_5: float
    pm97_5: float

    t50: float
    t10: float
    t90: float
    t02_5: float
    t97_5: float


@dataclass(frozen=True)
class EnsembleResult:
    n: int
    seed: int
    bands: List[BandsHour]
    pm_threshold: float
    heat_threshold_c: float


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """
    Deterministic quantile with linear interpolation.
    sorted_vals must be non-empty and sorted.
    q in [0, 1].
    """
    if not sorted_vals:
        raise ValueError("empty values")
    q = min(1.0, max(0.0, float(q)))
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])

    pos = q * (n - 1)
    lo = int(pos)
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _simulate_with_params(
    forcing: List[ForcingHour],
    building: BuildingProfile,
    pm_params: PMModelParams,
    th_params: ThermalParams,
    *,
    pm25_init: float,
    temp_init_c: float,
) -> List[OutputHour]:
    """
    Runs the same policy as simulate_demo(), but with explicit parameters.
    Keeps this self-contained so the ensemble doesn't depend on engine internals.
    """
    pm25_in = float(max(0.0, pm25_init))
    t_air = float(temp_init_c)
    t_mass = float(temp_init_c)
    dt_h = 1.0

    outputs: List[OutputHour] = []

    for f in forcing:
        # Use the same heuristic policy from the engine for now
        windows_open, hepa_on, fan_on, notes = sim_engine.choose_actions_for_hour(
            f=f,
            pm25_in_prev=pm25_in,
            temp_in_prev=t_air,
            building=building,
        )

        # PM step
        pm25_in = pm_step_with_controls(
            params=pm_params,
            c_prev=pm25_in,
            c_out=f.pm25_out,
            dt_h=dt_h,
            windows_open=windows_open,
            hepa_on=hepa_on,
            indoor_source_ug_h=0.0,
        )

        # Thermal step (scheduled internal gains)
        q_int_w = internal_gains_w(building.archetype, building.occupants, f.t)
        t_air, t_mass = thermal_step_2r2c(
            params=th_params,
            t_air_prev_c=t_air,
            t_mass_prev_c=t_mass,
            t_out_c=f.temp_out_c,
            dt_h=dt_h,
            windows_open=windows_open,
            q_internal_w=q_int_w,
            substeps=12,
        )

        outputs.append(OutputHour(t=f.t, pm25_in=pm25_in, temp_in_c=t_air))

    return outputs


def run_ensemble(
    forcing: List[ForcingHour],
    building: BuildingProfile,
    *,
    n: int = 200,
    seed: int = 123,
    priors: Priors | None = None,
    pm25_init: float = 8.0,
    temp_init_c: float = 28.5,
    pm_threshold: float = sim_engine.WHO_PM25_24H_GUIDELINE_UG_M3,
    heat_threshold_c: float = sim_engine.HEAT_THRESHOLD_C,
) -> EnsembleResult:
    """
    Run an ensemble of simulations and compute per-hour quantile bands.

    Returns:
    - bands per hour for PM and indoor temperature
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not forcing:
        raise ValueError("forcing is empty")

    priors = priors or Priors.defaults()

    # Collect samples per hour
    horizon = len(forcing)
    pm_samples: List[List[float]] = [[] for _ in range(horizon)]
    t_samples: List[List[float]] = [[] for _ in range(horizon)]

    for i in range(int(n)):
        pm_params, th_params = sample_joint_params(building, seed=int(seed) + i, priors=priors)
        outs = _simulate_with_params(
            forcing=forcing,
            building=building,
            pm_params=pm_params,
            th_params=th_params,
            pm25_init=pm25_init,
            temp_init_c=temp_init_c,
        )
        for h, o in enumerate(outs):
            pm_samples[h].append(float(o.pm25_in))
            t_samples[h].append(float(o.temp_in_c))

    bands: List[BandsHour] = []
    for h in range(horizon):
        pm_sorted = sorted(pm_samples[h])
        t_sorted = sorted(t_samples[h])

        bands.append(
            BandsHour(
                t=forcing[h].t,
                pm50=_quantile(pm_sorted, 0.50),
                pm10=_quantile(pm_sorted, 0.10),
                pm90=_quantile(pm_sorted, 0.90),
                pm02_5=_quantile(pm_sorted, 0.025),
                pm97_5=_quantile(pm_sorted, 0.975),
                t50=_quantile(t_sorted, 0.50),
                t10=_quantile(t_sorted, 0.10),
                t90=_quantile(t_sorted, 0.90),
                t02_5=_quantile(t_sorted, 0.025),
                t97_5=_quantile(t_sorted, 0.975),
            )
        )

    return EnsembleResult(
        n=int(n),
        seed=int(seed),
        bands=bands,
        pm_threshold=float(pm_threshold),
        heat_threshold_c=float(heat_threshold_c),
    )
