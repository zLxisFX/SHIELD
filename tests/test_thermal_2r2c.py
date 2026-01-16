from shield.models.thermal_2r2c import make_params, thermal_step_2r2c


def test_thermal_moves_toward_outdoor():
    params = make_params("classroom", floor_area_m2=90.0)

    t_air, t_mass = 30.0, 30.0
    t_out = 20.0

    t_air2, t_mass2 = thermal_step_2r2c(
        params=params,
        t_air_prev_c=t_air,
        t_mass_prev_c=t_mass,
        t_out_c=t_out,
        dt_h=1.0,
        windows_open=False,
        q_internal_w=0.0,
    )

    assert t_air2 < t_air  # should cool toward outdoor


def test_open_windows_cools_faster_than_closed():
    params = make_params("classroom", floor_area_m2=90.0)

    t_air, t_mass = 32.0, 32.0
    t_out = 24.0

    closed_air, _ = thermal_step_2r2c(
        params=params, t_air_prev_c=t_air, t_mass_prev_c=t_mass, t_out_c=t_out, dt_h=1.0, windows_open=False, q_internal_w=0.0
    )
    open_air, _ = thermal_step_2r2c(
        params=params, t_air_prev_c=t_air, t_mass_prev_c=t_mass, t_out_c=t_out, dt_h=1.0, windows_open=True, q_internal_w=0.0
    )

    assert open_air < closed_air
