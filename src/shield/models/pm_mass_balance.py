"""
Indoor PM2.5 mass-balance model.

Key design goal:
- Windows open vs closed MUST change the effective ACH and penetration.
- HEPA on/off MUST change the effective CADR removal.

Also includes compatibility aliases so other modules can safely access either
(old) or (new) attribute names.

IMPORTANT: make_params() is backwards-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from shield.core.state import BuildingProfile


# -----------------------------
# Defaults
# -----------------------------

@dataclass(frozen=True)
class PMArchetypeDefaults:
    volume_per_m2_m3: float
    ach_closed: float
    ach_open: float
    pen_closed: float
    pen_open: float
    k_dep_h: float
    cadr_hepa_per_m2_m3_h: float  # CADR scales with floor area


DEFAULTS: Dict[str, PMArchetypeDefaults] = {
    "apartment": PMArchetypeDefaults(
        volume_per_m2_m3=2.7,
        ach_closed=0.6,
        ach_open=3.0,
        pen_closed=0.30,
        pen_open=0.95,
        k_dep_h=0.18,
        cadr_hepa_per_m2_m3_h=4.0,
    ),
    "house": PMArchetypeDefaults(
        volume_per_m2_m3=3.0,
        ach_closed=0.7,
        ach_open=3.5,
        pen_closed=0.35,
        pen_open=0.95,
        k_dep_h=0.18,
        cadr_hepa_per_m2_m3_h=4.5,
    ),
    "classroom": PMArchetypeDefaults(
        volume_per_m2_m3=3.0,
        ach_closed=0.8,
        ach_open=4.0,
        pen_closed=0.35,
        pen_open=0.95,
        k_dep_h=0.18,
        cadr_hepa_per_m2_m3_h=6.0,
    ),
    "clinic": PMArchetypeDefaults(
        volume_per_m2_m3=2.8,
        ach_closed=0.9,
        ach_open=4.0,
        pen_closed=0.35,
        pen_open=0.95,
        k_dep_h=0.18,
        cadr_hepa_per_m2_m3_h=6.5,
    ),
}


# -----------------------------
# Parameter dataclass
# -----------------------------

@dataclass(frozen=True)
class PMModelParams:
    archetype: str
    floor_area_m2: float
    volume_m3: float

    ach_closed: float
    ach_open: float

    pen_closed: float
    pen_open: float

    k_dep_h: float
    cadr_hepa_m3_h: float

    # ---- Compatibility aliases (older names used earlier in the project) ----
    @property
    def ach_closed_per_h(self) -> float:
        return self.ach_closed

    @property
    def ach_open_per_h(self) -> float:
        return self.ach_open

    @property
    def penetration_closed(self) -> float:
        return self.pen_closed

    @property
    def penetration_open(self) -> float:
        return self.pen_open

    @property
    def deposition_k_per_h(self) -> float:
        return self.k_dep_h

    @property
    def hepa_cadr_m3_h(self) -> float:
        return self.cadr_hepa_m3_h


def defaults_for_archetype(archetype: str, floor_area_m2: float, has_hepa: bool = True) -> PMModelParams:
    a = str(archetype).lower()
    if a not in DEFAULTS:
        a = "classroom"

    d = DEFAULTS[a]
    area = float(floor_area_m2)
    volume = float(d.volume_per_m2_m3 * area)

    cadr = float(d.cadr_hepa_per_m2_m3_h * area) if bool(has_hepa) else 0.0

    return PMModelParams(
        archetype=a,
        floor_area_m2=area,
        volume_m3=volume,
        ach_closed=float(d.ach_closed),
        ach_open=float(d.ach_open),
        pen_closed=float(d.pen_closed),
        pen_open=float(d.pen_open),
        k_dep_h=float(d.k_dep_h),
        cadr_hepa_m3_h=cadr,
    )


def make_params(*args: Any, **kwargs: Any) -> PMModelParams:
    """
    Backwards-compatible factory.

    Supports:
      - make_params(building: BuildingProfile)
      - make_params(archetype: str, floor_area_m2: float)
      - make_params(archetype: str, floor_area_m2: float, has_hepa: bool)
      - make_params("classroom", floor_area_m2=90.0)
      - make_params(archetype="classroom", floor_area_m2=90.0, has_hepa=True)
    """
    # Style 1: make_params(building)
    if len(args) >= 1 and isinstance(args[0], BuildingProfile):
        b: BuildingProfile = args[0]
        return defaults_for_archetype(b.archetype, float(b.floor_area_m2), bool(b.has_hepa))

    # Style 2: positional archetype, floor_area_m2
    has_hepa = kwargs.pop("has_hepa", True)

    archetype = args[0] if len(args) >= 1 else kwargs.pop("archetype", "classroom")

    if len(args) >= 2:
        floor_area_m2 = args[1]
    else:
        if "floor_area_m2" in kwargs:
            floor_area_m2 = kwargs.pop("floor_area_m2")
        elif "area" in kwargs:
            floor_area_m2 = kwargs.pop("area")
        else:
            raise TypeError("make_params() missing required argument: floor_area_m2")

    # If anything unexpected remains, fail loudly (helps catch bugs)
    if kwargs:
        raise TypeError(f"make_params() got unexpected keyword arguments: {sorted(kwargs.keys())}")

    return defaults_for_archetype(str(archetype), float(floor_area_m2), bool(has_hepa))


# -----------------------------
# Dynamics
# -----------------------------

def mass_balance_step_exact(
    *,
    c_prev: float,
    c_out: float,
    dt_h: float,
    ach_h: float,
    penetration: float,
    k_dep_h: float,
    cadr_m3_h: float,
    volume_m3: float,
    indoor_source_ug_h: float = 0.0,
) -> float:
    """
    Exact 1st-order solution for:
      dC/dt = ach*pen*Cout + S/V - (ach + kdep + CADR/V)*C
    """
    c_prev = float(c_prev)
    c_out = float(c_out)
    dt_h = float(dt_h)

    ach_h = max(0.0, float(ach_h))
    penetration = max(0.0, min(1.0, float(penetration)))
    k_dep_h = max(0.0, float(k_dep_h))
    cadr_m3_h = max(0.0, float(cadr_m3_h))
    volume_m3 = max(1e-6, float(volume_m3))
    indoor_source_ug_h = max(0.0, float(indoor_source_ug_h))

    lam = ach_h + k_dep_h + (cadr_m3_h / volume_m3)
    b = (ach_h * penetration * c_out) + (indoor_source_ug_h / volume_m3)

    if lam <= 0.0:
        c_next = c_prev + b * dt_h
    else:
        import math

        decay = math.exp(-lam * dt_h)
        c_next = (c_prev * decay) + (b / lam) * (1.0 - decay)

    return max(0.0, float(c_next))


def pm_step_with_controls(
    *,
    params: PMModelParams,
    c_prev: float,
    c_out: float,
    dt_h: float,
    windows_open: bool,
    hepa_on: bool,
    indoor_source_ug_h: float = 0.0,
) -> float:
    """
    One-hour PM step that correctly switches between OPEN/CLOSED ventilation regimes.
    """
    if bool(windows_open):
        ach = params.ach_open
        pen = params.pen_open
    else:
        ach = params.ach_closed
        pen = params.pen_closed

    cadr = params.cadr_hepa_m3_h if bool(hepa_on) else 0.0

    return mass_balance_step_exact(
        c_prev=float(c_prev),
        c_out=float(c_out),
        dt_h=float(dt_h),
        ach_h=float(ach),
        penetration=float(pen),
        k_dep_h=float(params.k_dep_h),
        cadr_m3_h=float(cadr),
        volume_m3=float(params.volume_m3),
        indoor_source_ug_h=float(indoor_source_ug_h),
    )
