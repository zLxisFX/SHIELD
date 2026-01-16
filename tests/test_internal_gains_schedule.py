from datetime import datetime

from shield.models.thermal_2r2c import internal_gains_w


def test_classroom_gains_low_after_school():
    # Jan 15 2026 is a Thursday (weekday)
    t = datetime(2026, 1, 15, 17, 0)
    q = internal_gains_w("classroom", occupants=25, t=t)
    assert q < 300.0  # should be near idle load


def test_classroom_gains_high_during_school():
    t = datetime(2026, 1, 15, 10, 0)
    q = internal_gains_w("classroom", occupants=25, t=t)
    assert q > 1000.0  # occupied
