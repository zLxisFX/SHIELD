"""
Indoor PM source models (simple, offline, deterministic).

Purpose:
- Create realistic indoor particle generation that depends on archetype + time.
- This creates situations where outdoor-based heuristics fail, but planning wins.

Units:
- Returns indoor_source_ug_h (micrograms per hour) used by pm_step_with_controls().
"""

from __future__ import annotations

from datetime import datetime


def indoor_source_ug_h(archetype: str, occupants: int, t: datetime) -> float:
    a = str(archetype).lower()
    hour = int(t.hour)
    occ = max(0, int(occupants))

    source = 0.0

    if a == "classroom":
        # Occupied hours resuspension baseline
        # (movement, dust, chairs, bags, hallway traffic)
        if 8 <= hour <= 15:
            source += 400.0 * occ  # stronger than before to make the effect visible

        # Short spikes that a reactive outdoor rule WILL miss
        # (class change / recess / lunch rush)
        if hour in (9, 10, 11):      # morning transitions
            source += 25000.0
        if hour in (12, 13):         # lunch transition
            source += 18000.0

        # Cleaning pulse after school
        if hour == 16:
            source += 30000.0
        if hour == 17:
            source += 18000.0

    elif a == "clinic":
        if 7 <= hour <= 18:
            source += 180.0 * occ
        if hour in (19,):
            source += 6000.0

    elif a in ("house", "apartment"):
        if hour in (7, 8):
            source += 9000.0
        if hour in (18, 19, 20):
            source += 12000.0

    return float(max(0.0, source))
