from datetime import datetime

from shield.core.state import BuildingProfile
from shield.demo.scenarios import make_demo_forcing
from shield.engine.simulate import simulate_demo


def test_simulate_demo_produces_runresult():
    building = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)
    forcing = make_demo_forcing(start=datetime.now(), horizon_hours=6, scenario="smoke_heat_day")

    run = simulate_demo(forcing=forcing, building=building)

    assert run.horizon_hours == 6
    assert len(run.forcing) == 6
    assert len(run.actions) == 6
    assert len(run.outputs) == 6
    assert run.pm25_threshold > 0
    assert run.heat_threshold_c > 0
