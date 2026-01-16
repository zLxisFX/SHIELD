from shield.models.pm_mass_balance import PMModelParams, pm_step_with_controls


def test_windows_open_increases_indoor_pm_when_outdoor_high():
    p = PMModelParams(
        archetype="classroom",
        floor_area_m2=90.0,
        volume_m3=270.0,
        ach_closed=0.8,
        ach_open=4.0,
        pen_closed=0.35,
        pen_open=0.95,
        k_dep_h=0.0,
        cadr_hepa_m3_h=0.0,
    )

    c_prev = 0.0
    c_out = 150.0  # smoky outdoor
    dt_h = 1.0

    c_closed = pm_step_with_controls(
        params=p,
        c_prev=c_prev,
        c_out=c_out,
        dt_h=dt_h,
        windows_open=False,
        hepa_on=False,
        indoor_source_ug_h=0.0,
    )
    c_open = pm_step_with_controls(
        params=p,
        c_prev=c_prev,
        c_out=c_out,
        dt_h=dt_h,
        windows_open=True,
        hepa_on=False,
        indoor_source_ug_h=0.0,
    )

    assert c_open > c_closed, "Opening windows must increase indoor PM when outdoor is high."
    assert c_open > 2.0 * c_closed, "Effect should be substantial for high outdoor PM."
