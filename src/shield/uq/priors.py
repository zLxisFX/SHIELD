"""
Uncertainty Quantification (UQ) priors + parameter sampling for SHIELD.

Goal:
- Provide deterministic, offline-friendly parameter priors (no external deps)
- Sample "plausible" physics parameters around archetype defaults
- Enable ensemble runs: median + 80/95% intervals
- Later: swap priors for calibrated posteriors using sensor data (Bayesian update)

Design:
- Use multiplicative lognormal factors for positive parameters (ACH, CADR, resistances, capacities, loss rates)
- Use truncated normal factors for bounded parameters (penetration 0..1)
- Keep sampling reproducible via an explicit seed
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

from shield.core.state import BuildingProfile

from shield.models.pm_mass_balance import PMModelParams, make_params as make_pm_defaults
from shield.models.thermal_2r2c import ThermalParams, make_params as make_th_defaults


# -------------------------
# Helper distributions
# -------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _lognormal_factor(rng: random.Random, sigma: float, *, min_mult: float, max_mult: float) -> float:
    """
    Multiplicative factor ~ LogNormal(0, sigma) but clamped to [min_mult, max_mult].
    """
    # normalvariate returns N(mu, sigma)
    z = rng.normalvariate(0.0, float(max(0.0, sigma)))
    f = math.exp(z)
    return float(_clamp(f, min_mult, max_mult))


def _trunc_normal(rng: random.Random, mu: float, sigma: float, lo: float, hi: float) -> float:
    """
    Simple rejection sampling truncated normal.
    Safe for small truncation ranges used here.
    """
    mu = float(mu)
    sigma = float(max(1e-9, sigma))
    lo = float(lo)
    hi = float(hi)
    for _ in range(200):
        x = rng.normalvariate(mu, sigma)
        if lo <= x <= hi:
            return float(x)
    # fallback: clamp mean
    return float(_clamp(mu, lo, hi))


# -------------------------
# Prior specifications
# -------------------------

@dataclass(frozen=True)
class LogNormalPrior:
    """
    Prior over multiplicative factor f (positive).
    If base parameter is x0, sampled x = x0 * f.
    """
    sigma: float = 0.35
    min_mult: float = 0.30
    max_mult: float = 3.00

    def sample_factor(self, rng: random.Random) -> float:
        return _lognormal_factor(rng, self.sigma, min_mult=self.min_mult, max_mult=self.max_mult)


@dataclass(frozen=True)
class PenetrationPrior:
    """
    Penetration is bounded [0, 1].
    We sample around the base value using a truncated normal.
    """
    sigma_abs: float = 0.08  # absolute (not multiplicative)

    def sample(self, rng: random.Random, base: float) -> float:
        base = float(_clamp(base, 0.0, 1.0))
        return _trunc_normal(rng, mu=base, sigma=self.sigma_abs, lo=0.0, hi=1.0)


@dataclass(frozen=True)
class Priors:
    """
    Bundle of priors used to sample model parameters.
    """

    # PM model
    ach: LogNormalPrior
    k_dep: LogNormalPrior
    cadr: LogNormalPrior
    penetration: PenetrationPrior

    # Thermal model
    r_ao: LogNormalPrior
    r_am: LogNormalPrior
    c_air: LogNormalPrior
    c_mass: LogNormalPrior

    @staticmethod
    def defaults() -> "Priors":
        """
        Conservative default priors (offline demo safe).
        We will later tune these using literature and calibration.
        """
        return Priors(
            ach=LogNormalPrior(sigma=0.45, min_mult=0.25, max_mult=4.0),
            k_dep=LogNormalPrior(sigma=0.35, min_mult=0.40, max_mult=2.5),
            cadr=LogNormalPrior(sigma=0.50, min_mult=0.20, max_mult=5.0),
            penetration=PenetrationPrior(sigma_abs=0.07),
            r_ao=LogNormalPrior(sigma=0.35, min_mult=0.40, max_mult=2.5),
            r_am=LogNormalPrior(sigma=0.35, min_mult=0.40, max_mult=2.5),
            c_air=LogNormalPrior(sigma=0.25, min_mult=0.60, max_mult=1.8),
            c_mass=LogNormalPrior(sigma=0.35, min_mult=0.40, max_mult=2.5),
        )


def seeded_rng(seed: int) -> random.Random:
    """
    Create a reproducible RNG.
    """
    return random.Random(int(seed))


# -------------------------
# Sampling functions
# -------------------------

def sample_pm_params(
    base: PMModelParams,
    priors: Priors,
    rng: random.Random,
) -> PMModelParams:
    """
    Sample PM parameters around deterministic archetype defaults.
    """
    f_ach_closed = priors.ach.sample_factor(rng)
    f_ach_open = priors.ach.sample_factor(rng)

    f_kdep = priors.k_dep.sample_factor(rng)
    f_cadr = priors.cadr.sample_factor(rng)

    pen_closed = priors.penetration.sample(rng, base.pen_closed)
    pen_open = priors.penetration.sample(rng, base.pen_open)

    return PMModelParams(
        archetype=base.archetype,
        floor_area_m2=base.floor_area_m2,
        volume_m3=base.volume_m3,
        ach_closed=float(max(0.0, base.ach_closed * f_ach_closed)),
        ach_open=float(max(0.0, base.ach_open * f_ach_open)),
        pen_closed=float(_clamp(pen_closed, 0.0, 1.0)),
        pen_open=float(_clamp(pen_open, 0.0, 1.0)),
        k_dep_h=float(max(0.0, base.k_dep_h * f_kdep)),
        cadr_hepa_m3_h=float(max(0.0, base.cadr_hepa_m3_h * f_cadr)),
    )


def sample_thermal_params(
    base: ThermalParams,
    priors: Priors,
    rng: random.Random,
) -> ThermalParams:
    """
    Sample thermal parameters around deterministic archetype defaults.
    """
    f_r_ao = priors.r_ao.sample_factor(rng)
    f_r_am = priors.r_am.sample_factor(rng)
    f_c_air = priors.c_air.sample_factor(rng)
    f_c_mass = priors.c_mass.sample_factor(rng)

    # We also allow ACH uncertainty through same ach prior
    f_ach_closed = priors.ach.sample_factor(rng)
    f_ach_open = priors.ach.sample_factor(rng)

    return ThermalParams(
        archetype=base.archetype,
        floor_area_m2=base.floor_area_m2,
        volume_m3=base.volume_m3,
        r_ao_k_w=float(max(1e-6, base.r_ao_k_w * f_r_ao)),
        r_am_k_w=float(max(1e-6, base.r_am_k_w * f_r_am)),
        c_air_j_k=float(max(1e3, base.c_air_j_k * f_c_air)),
        c_mass_j_k=float(max(1e3, base.c_mass_j_k * f_c_mass)),
        ach_closed=float(max(0.0, base.ach_closed * f_ach_closed)),
        ach_open=float(max(0.0, base.ach_open * f_ach_open)),
    )


def sample_joint_params(
    building: BuildingProfile,
    *,
    seed: int,
    priors: Priors | None = None,
) -> Tuple[PMModelParams, ThermalParams]:
    """
    Convenience: create defaults from building + sample both PM and Thermal params.
    """
    priors = priors or Priors.defaults()
    rng = seeded_rng(seed)

    base_pm = make_pm_defaults(building.archetype, building.floor_area_m2)
    base_th = make_th_defaults(building.archetype, building.floor_area_m2)

    pm = sample_pm_params(base_pm, priors, rng)
    th = sample_thermal_params(base_th, priors, rng)
    return pm, th


def priors_summary_dict(priors: Priors | None = None) -> Dict[str, float]:
    """
    Handy for model cards / debug printing.
    """
    p = priors or Priors.defaults()
    return {
        "ach_sigma": p.ach.sigma,
        "k_dep_sigma": p.k_dep.sigma,
        "cadr_sigma": p.cadr.sigma,
        "pen_sigma_abs": p.penetration.sigma_abs,
        "r_ao_sigma": p.r_ao.sigma,
        "r_am_sigma": p.r_am.sigma,
        "c_air_sigma": p.c_air.sigma,
        "c_mass_sigma": p.c_mass.sigma,
    }
