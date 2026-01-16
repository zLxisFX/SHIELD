from datetime import datetime
from shield.demo.scenarios import make_demo_forcing

start = datetime(2026, 1, 15, 0, 0)
forcing = make_demo_forcing(start=start, horizon_hours=72, scenario="smoke_heat_day")

print("t | pm25_out | temp_out_c")
print("-" * 50)
for f in forcing:
    if f.t.hour in (9, 10, 11, 12, 13, 14):
        print(f"{f.t.isoformat(timespec='minutes')} | {f.pm25_out:7.1f} | {f.temp_out_c:9.1f}")