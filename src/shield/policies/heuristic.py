"""
Heuristic policy baseline.

Separated into its own module to avoid circular imports:
- engine.simulate imports heuristic for default behavior
- optimizer.robust_greedy imports heuristic for lookahead rollouts
"""

from __future__ import annotations

from typing import List, Tuple

from shield.core.state import BuildingProfile, ForcingHour

# ---- Heuristic thresholds (simple, explainable) ----
SMOKE_HIGH_OUT_UG_M3 = 35.0
SMOKE_LOW_OUT_UG_M3 = 12.0
SMOKE_HEPA_ON_OUT_UG_M3 = 15.0

HOT_IN_C = 30.5
HOT_OUT_C = 30.0
FAN_ON_IN_C = 29.0

# Only ventilate for cooling if outdoor is meaningfully cooler than indoor.
# This prevents opening windows when it's hotter outside (which can blow up heat risk).
COOLING_DELTA_C = 0.5  # outside must be at least 0.5°C cooler than indoor


def choose_actions_for_hour(
    f: ForcingHour,
    pm25_in_prev: float,
    temp_in_prev: float,
    building: BuildingProfile,
) -> Tuple[bool, bool, bool, List[str]]:
    """
    Baseline decision rules:
    - close windows when smoke high
    - ventilate ONLY when smoke low AND indoor hot AND outdoor cooler than indoor
    - run HEPA when smoke moderate/high (if available)
    - run fan when hot (if available)
    """
    notes: List[str] = []

    # --- Smoke bands ---
    smoke_high = f.pm25_out >= SMOKE_HIGH_OUT_UG_M3
    smoke_low = f.pm25_out <= SMOKE_LOW_OUT_UG_M3

    # --- Heat bands ---
    hot_in = temp_in_prev >= HOT_IN_C
    hot_out = f.temp_out_c >= HOT_OUT_C

    # --- Filtration ---
    hepa_on = bool(building.has_hepa and (f.pm25_out >= SMOKE_HEPA_ON_OUT_UG_M3))
    if hepa_on:
        notes.append("Run HEPA / filter")

    # --- Fan (comfort / mixing) ---
    fan_on = bool(building.has_fan and (temp_in_prev >= FAN_ON_IN_C or hot_out))
    if fan_on:
        notes.append("Run fan for cooling/comfort")

    # --- Windows decision ---
    windows_open = False

    if smoke_high:
        windows_open = False
        notes.append("Keep windows CLOSED (smoke)")
    else:
        # Only open for cooling if:
        # 1) outside smoke is low enough, and
        # 2) indoor is hot, and
        # 3) outside is actually cooler than inside by COOLING_DELTA_C
        outdoor_cools = f.temp_out_c <= (temp_in_prev - COOLING_DELTA_C)

        if smoke_low and hot_in and outdoor_cools:
            windows_open = True
            notes.append("Ventilate (windows OPEN) for cooling")
        else:
            windows_open = False
            if smoke_low and hot_in and (not outdoor_cools):
                notes.append("Outside not cooler; keep windows mostly CLOSED")
            else:
                notes.append("Windows mostly CLOSED (balance)")

    # Advice when no HEPA during smoky periods
    if (not building.has_hepa) and (f.pm25_out >= SMOKE_HEPA_ON_OUT_UG_M3):
        notes.append("Seal gaps / use towel draft blockers (no HEPA)")

    return windows_open, hepa_on, fan_on, notes
