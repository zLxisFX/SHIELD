from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Literal

Mode = Literal["demo", "live"]
Archetype = Literal["apartment", "house", "classroom", "clinic"]


@dataclass(frozen=True)
class BuildingProfile:
    archetype: Archetype
    floor_area_m2: float
    has_hepa: bool = True
    has_fan: bool = True
    occupants: int = 1


@dataclass(frozen=True)
class ForcingHour:
    t: datetime
    pm25_out: float
    temp_out_c: float
    rh_out: float


@dataclass(frozen=True)
class ActionsHour:
    t: datetime
    windows_open: bool
    hepa_on: bool
    fan_on: bool
    notes: List[str]


@dataclass(frozen=True)
class OutputHour:
    t: datetime
    pm25_in: float
    temp_in_c: float


@dataclass(frozen=True)
class RunResult:
    start: datetime
    horizon_hours: int
    building: BuildingProfile
    forcing: List[ForcingHour]
    actions: List[ActionsHour]
    outputs: List[OutputHour]
    minutes_pm25_above_threshold: int
    hours_heat_above_threshold: int
    pm25_threshold: float
    heat_threshold_c: float
    warnings: List[str]


def make_time_grid(start: datetime, horizon_hours: int) -> List[datetime]:
    start0 = start.replace(minute=0, second=0, microsecond=0)
    return [start0 + timedelta(hours=h) for h in range(horizon_hours)]


def validate_forcing(forcing: List[ForcingHour], horizon_hours: int) -> None:
    if len(forcing) != horizon_hours:
        raise ValueError(f"Forcing length {len(forcing)} != horizon_hours {horizon_hours}")
    for f in forcing:
        if not (0.0 <= f.rh_out <= 1.0):
            raise ValueError(f"rh_out must be 0..1, got {f.rh_out}")
        if f.pm25_out < 0:
            raise ValueError(f"pm25_out must be >=0, got {f.pm25_out}")


def validate_actions(actions: List[ActionsHour], horizon_hours: int) -> None:
    if len(actions) != horizon_hours:
        raise ValueError(f"Actions length {len(actions)} != horizon_hours {horizon_hours}")


def validate_outputs(outputs: List[OutputHour], horizon_hours: int) -> None:
    if len(outputs) != horizon_hours:
        raise ValueError(f"Outputs length {len(outputs)} != horizon_hours {horizon_hours}")
