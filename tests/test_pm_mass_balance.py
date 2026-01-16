import math

from shield.models.pm_mass_balance import mass_balance_step_exact, make_params, pm_step_with_controls


def test_exact_step_converges_to_outdoor_when_only_ventilation():
    # If only ventilation exists (no deposition, no filtration), indoor should approach p*Cout
    c_prev = 0.0
    c_out = 100.0
    dt_h = 10.0

    c_next = mass_balance_step_exact(
        c_prev=c_prev,
        c_out=c_out,
        dt_h=dt_h,
        ach_h=1.0,
        penetration=1.0,
        k_dep_h=0.0,
        cadr_m3_h=0.0,
        volume_m3=100.0,
        indoor_source_ug_h=0.0,
    )

    # closed form: C = Cout*(1 - exp(-a*dt))
    expected = c_out * (1.0 - math.exp(-1.0 * dt_h))
    assert abs(c_next - expected) < 1e-6


def test_filtration_reduces_concentration():
    # With strong filtration, indoor should be much lower than without filtration
    c_prev = 50.0
    c_out = 100.0
    dt_h = 1.0

    no_filter = mass_balance_step_exact(
        c_prev=c_prev,
        c_out=c_out,
        dt_h=dt_h,
        ach_h=0.5,
        penetration=1.0,
        k_dep_h=0.0,
        cadr_m3_h=0.0,
        volume_m3=100.0,
        indoor_source_ug_h=0.0,
    )

    with_filter = mass_balance_step_exact(
        c_prev=c_prev,
        c_out=c_out,
        dt_h=dt_h,
        ach_h=0.5,
        penetration=1.0,
        k_dep_h=0.0,
        cadr_m3_h=600.0,  # CADR 600 m3/h, volume 100 m3 => k_fil=6/h
        volume_m3=100.0,
        indoor_source_ug_h=0.0,
    )

    assert with_filter < no_filter


def test_pm_step_with_controls_runs_and_is_nonnegative():
    params = make_params("classroom", floor_area_m2=90.0)
    c = 0.0
    for _ in range(6):
        c = pm_step_with_controls(
            params=params,
            c_prev=c,
            c_out=120.0,
            dt_h=1.0,
            windows_open=False,
            hepa_on=True,
        )
        assert c >= 0.0
