"""
Baseline policies for SHIELD evaluation.

These baselines are deliberately simple and interpretable.
They let us compute decision-centric improvements:
- minutes above PM threshold
- hours above heat threshold

Baselines implemented:
1) always_closed: windows always closed; HEPA on if available; fan optional for comfort
2) always_open: windows always open (ventilation); HEPA off (typically pointless if windows open, but configurable)
3) night_vent: windows open at night hours (cooling), closed in day; HEPA on when closed if available
4) hepa_always: windows closed; HEPA always on if available

These provide "common sense" strategies people actually do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from shield.core.state import ActionsHour, BuildingProfile, ForcingHour
from shield.policies.heuristic import choose_actions_for_hour as heuristic_policy


BaselineName = Literal["heuristic", "always_closed", "always_open", "night_vent", "hepa_always"]


@dataclass(frozen=True)
class BaselineConfig:
    night_start: int = 20  # 8pm
    night_end: int = 6     # 6am
    fan_for_hot_outdoor: bool = True
    hepa_when_windows_open: bool = False  # usually not useful, but allow for experiments


def _fan_rule(f: ForcingHour, building: BuildingProfile, *, cfg: BaselineConfig) -> bool:
    if not building.has_fan:
        return False
    if cfg.fan_for_hot_outdoor and f.temp_out_c >= 30.0:
        return True
    return False


def baseline_actions(
    forcing: List[ForcingHour],
    building: BuildingProfile,
    name: BaselineName,
    *,
    cfg: Optional[BaselineConfig] = None,
) -> List[ActionsHour]:
    """
    Create an action schedule for a named baseline policy.
    """
    if not forcing:
        raise ValueError("forcing empty")

    cfg = cfg or BaselineConfig()
    actions: List[ActionsHour] = []

    # For heuristic baseline, we must simulate statefully; but this file is "schedule builder".
    # We'll implement a lightweight state rollout (PM/T) just to call the heuristic.
    # This keeps baselines comparable and avoids importing engine (no circular imports).
    pm_prev = 8.0
    t_prev = 28.5

    for f in forcing:
        if name == "heuristic":
            w, h, fan, notes = heuristic_policy(f, pm_prev, t_prev, building)
            # keep state proxies simple; the true physics is evaluated in engine anyway
            pm_prev = pm_prev * 0.9 + f.pm25_out * (0.1 if not w else 0.2)
            t_prev = t_prev * 0.9 + f.temp_out_c * (0.1 if not w else 0.2)
            actions.append(ActionsHour(t=f.t, windows_open=w, hepa_on=h, fan_on=fan, notes=notes))
            continue

        if name == "always_closed":
            windows_open = False
            hepa_on = building.has_hepa
            fan_on = _fan_rule(f, building, cfg=cfg)
            notes = []
            if hepa_on:
                notes.append("Run HEPA / filter (baseline: always closed)")
            if fan_on:
                notes.append("Run fan for cooling/comfort (baseline)")
            notes.append("Keep windows CLOSED (baseline)")
            actions.append(ActionsHour(t=f.t, windows_open=windows_open, hepa_on=hepa_on, fan_on=fan_on, notes=notes))
            continue

        if name == "always_open":
            windows_open = True
            hepa_on = building.has_hepa and bool(cfg.hepa_when_windows_open)
            fan_on = _fan_rule(f, building, cfg=cfg)
            notes = []
            if fan_on:
                notes.append("Run fan for cooling/comfort (baseline)")
            if hepa_on:
                notes.append("Run HEPA / filter (baseline; windows open)")
            notes.append("Ventilate (windows OPEN) (baseline)")
            actions.append(ActionsHour(t=f.t, windows_open=windows_open, hepa_on=hepa_on, fan_on=fan_on, notes=notes))
            continue

        if name == "night_vent":
            hour = f.t.hour
            is_night = (hour >= cfg.night_start) or (hour < cfg.night_end)
            windows_open = bool(is_night)
            hepa_on = building.has_hepa and (not windows_open)
            fan_on = _fan_rule(f, building, cfg=cfg)
            notes = []
            if windows_open:
                notes.append("Ventilate (windows OPEN) (baseline: night vent)")
            else:
                notes.append("Keep windows CLOSED (baseline: night vent)")
                if hepa_on:
                    notes.append("Run HEPA / filter (baseline: night vent)")
            if fan_on:
                notes.append("Run fan for cooling/comfort (baseline)")
            actions.append(ActionsHour(t=f.t, windows_open=windows_open, hepa_on=hepa_on, fan_on=fan_on, notes=notes))
            continue

        if name == "hepa_always":
            windows_open = False
            hepa_on = building.has_hepa
            fan_on = _fan_rule(f, building, cfg=cfg)
            notes = []
            if hepa_on:
                notes.append("Run HEPA / filter (baseline: always on)")
            if fan_on:
                notes.append("Run fan for cooling/comfort (baseline)")
            notes.append("Keep windows CLOSED (baseline)")
            actions.append(ActionsHour(t=f.t, windows_open=windows_open, hepa_on=hepa_on, fan_on=fan_on, notes=notes))
            continue

        raise ValueError(f"Unknown baseline: {name}")

    return actions


def list_baselines() -> List[str]:
    return ["heuristic", "always_closed", "always_open", "night_vent", "hepa_always"]
