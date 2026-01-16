from datetime import datetime
from pathlib import Path

from shield.core.state import BuildingProfile
from shield.demo.scenarios import make_demo_forcing
from shield.demo.export import export_runresult
from shield.engine.simulate import simulate_demo


def test_demo_runs_and_exports(tmp_path: Path):
    building = BuildingProfile(archetype="classroom", floor_area_m2=90.0, has_hepa=True, has_fan=True, occupants=25)

    forcing = make_demo_forcing(start=datetime.now(), horizon_hours=6, scenario="smoke_heat_day")
    run = simulate_demo(forcing=forcing, building=building)

    assert run.horizon_hours == 6
    assert len(run.outputs) == 6
    assert len(run.actions) == 6

    txt_path, json_path = export_runresult(run, out_dir=tmp_path, prefix="test_plan")

    assert txt_path.exists()
    assert json_path.exists()
    assert txt_path.suffix == ".txt"
    assert json_path.suffix == ".json"
