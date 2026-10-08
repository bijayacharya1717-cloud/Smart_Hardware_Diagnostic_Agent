"""Generate a synthetic sensor log with one injected overheating fault.

Run from the project root:
    python sensor_tool/generate_sensor_logs.py

Writes Data/sensor_logs.csv (overwrites the existing file). Uses only the
standard library, and a fixed seed so every run produces identical data.

Scenario (1 reading per minute, 300 minutes):
  - Normal operation: temperature ~42 C, pressure ~101.3 kPa, humidity ~55 %
  - Fault: starting at 13:00 the temperature ramps up ~2 C/min, peaks above
    85 C (the BME280's upper operating limit; verify against your datasheet),
    holds for a few minutes, then cools back to normal.
  - Humidity dips slightly while the temperature is high (as it would
    physically). Pressure is unaffected.
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
START = datetime(2026, 10, 8, 10, 0, 0)
N_MINUTES = 300

# Fault timeline, in minutes after START (180 min = 13:00)
RAMP_START = 180
RAMP_MINUTES = 25      # ramp up at ~2 C/min
HOLD_MINUTES = 8       # stay at the peak
COOL_MINUTES = 20      # return to normal

BASE_TEMP = 42.5
PEAK_TEMP = 92.0

OUT_PATH = Path(__file__).resolve().parent.parent / "Data" / "sensor_logs.csv"


def noise(rng: random.Random, sd: float) -> float:
    """Gaussian sensor noise, clipped at +/-3 sd so random outliers can't
    masquerade as faults. The injected overheating is the only true anomaly."""
    return max(-3.0, min(3.0, rng.gauss(0, 1))) * sd


def fault_offset(minute: int) -> float:
    """Extra degrees C added on top of the normal baseline at this minute."""
    peak_rise = PEAK_TEMP - BASE_TEMP
    t = minute - RAMP_START
    if t < 0:
        return 0.0
    if t < RAMP_MINUTES:
        return peak_rise * t / RAMP_MINUTES
    t -= RAMP_MINUTES
    if t < HOLD_MINUTES:
        return peak_rise
    t -= HOLD_MINUTES
    if t < COOL_MINUTES:
        return peak_rise * (1 - t / COOL_MINUTES)
    return 0.0


def main() -> None:
    rng = random.Random(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["timestamp", "temperature", "pressure", "humidity"])
        for minute in range(N_MINUTES):
            ts = START + timedelta(minutes=minute)
            offset = fault_offset(minute)
            temp = BASE_TEMP + noise(rng, 0.3) + offset
            pressure = 101.3 + noise(rng, 0.08)
            humidity = 55.0 - 0.15 * offset + noise(rng, 0.4)
            writer.writerow([
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                round(temp, 2),
                round(pressure, 2),
                round(humidity, 2),
            ])

    peak_minute = RAMP_START + RAMP_MINUTES
    print(f"Wrote {N_MINUTES} rows to {OUT_PATH}")
    print(f"Fault starts at {START + timedelta(minutes=RAMP_START)}, "
          f"peaks at ~{PEAK_TEMP} C around {START + timedelta(minutes=peak_minute)}")


if __name__ == "__main__":
    main()
