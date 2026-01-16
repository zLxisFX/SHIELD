def test_core_and_demo_imports_work():
    # core imports
    from shield.core.state import BuildingProfile, make_time_grid
    assert BuildingProfile is not None
    assert len(make_time_grid.__name__) > 0

    # demo imports (this also verifies demo.scenarios can import shield.core.state)
    from shield.demo.scenarios import list_demo_scenarios, make_demo_forcing
    scenarios = list_demo_scenarios()
    assert "smoke_heat_day" in scenarios

    # quick forcing sanity
    forcing = make_demo_forcing(start=__import__("datetime").datetime.now(), horizon_hours=3, scenario="smoke_heat_day")
    assert len(forcing) == 3
