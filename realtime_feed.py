"""
GlacierGuard - Real-Time Data Ingestion Layer (v2: sim-clock + stable water model)
=====================================================================================
Fetches live rainfall/temperature (Open-Meteo) and seismic energy (USGS), and appends rows to
live_telemetry.csv in the SAME schema as synthetic_glacier_telemetry.csv.

What changed vs v1 (all found by reading your live log):
  1. TIMESTAMPS: v1 stamped every row with real time rounded DOWN to 10 min. run_demo ticks every
     60 s, so ~10 rows shared one timestamp -> dt=0 -> rate features forced to 0, and the model's
     row-count windows (5 / 20 rows) were fed 1-minute rows instead of the 10-minute cadence it
     was trained on. Fix: SIM_STEP_MIN. When set (run_demo sets 10), each row gets its own
     10-minute virtual timestamp (accelerated demo clock). Weather/seismic are still fetched with
     REAL time, so those inputs remain live. Leave SIM_STEP_MIN = None for real deployment
     (poll every 600 s).
  2. WATER LEVEL: v1 was a cumulative sum of rain*0.01 with no drain, so any rain made the level
     climb forever (about +0.029 m per row at 2.9 mm/hr). v2 is mean-reverting: the level relaxes
     toward a rain/temperature-dependent target. Still synthetic, still driven by real inputs.
  3. LAKE COORDINATES: v1 used (27.85, 88.5), roughly 30 km from the lake. Now taken from your
     lake_outline geojson (27.916, 88.209).
  4. Weather fetch failure (e.g. the 503 in your log) backs off for 5 minutes instead of retrying
     (and blocking the tick) every call; the last good cache is reused.
  5. USGS is queried once per real 10-minute slot, not every tick.

Run standalone (real cadence):   python realtime_feed.py
"""

import csv
import os
import time
import random
import requests
from datetime import datetime, timedelta, timezone

LAKE_LAT = 27.916          # from lake_outline_pre_breach_2022.geojson
LAKE_LON = 88.209
POLL_INTERVAL_SEC = 600    # 10 minutes = model's training cadence
OUTPUT_CSV = "live_telemetry.csv"
SEISMIC_WINDOW_MIN = 30

# None  -> row timestamp = real time (real deployment, POLL_INTERVAL_SEC = 600)
# 10    -> accelerated demo: every row is exactly 10 virtual minutes after the previous one
SIM_STEP_MIN = None

# water model (synthetic)
WATER_BASE_M = 2.55
WATER_RAIN_GAIN_M_PER_MM = 0.02
WATER_MELT_GAIN_M_PER_C = 0.004
WATER_RELAX = 0.15          # fraction of the gap to target closed per step

CSV_COLUMNS = [
    "timestamp_utc", "water_level_m", "vibration_rms_g", "tilt_deg", "rainfall_mm_hr",
    "turbidity_ntu", "ambient_temp_c", "seismic_energy",
    "water_valid", "vibration_valid", "tilt_valid", "rainfall_valid", "turbidity_valid",
    "seismic_valid", "sensor_status_water", "is_synthetic_water",
]

_weather_cache = {"fetched_at": None, "records": []}
_weather_retry_after = None
_seismic_seen_ids = set()
_history = []
_sim_clock = None
_last_seismic_slot = None


def round_to_10min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def reset_state():
    """Forget everything (call together with deleting live_telemetry.csv for a clean session)."""
    global _sim_clock, _last_seismic_slot, _weather_retry_after
    _history.clear()
    _seismic_seen_ids.clear()
    _sim_clock = None
    _last_seismic_slot = None
    _weather_retry_after = None


def fetch_weather(t_now: datetime):
    global _weather_retry_after
    fresh = (_weather_cache["fetched_at"] is not None
             and (t_now - _weather_cache["fetched_at"]) < timedelta(hours=1)
             and _weather_cache["records"])
    if fresh:
        return _weather_cache["records"]
    if _weather_retry_after is not None and t_now < _weather_retry_after:
        return _weather_cache["records"]

    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={LAKE_LAT}&longitude={LAKE_LON}"
           "&hourly=temperature_2m,precipitation&past_days=1")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in data["hourly"]["time"]]
        records = list(zip(times, data["hourly"]["temperature_2m"], data["hourly"]["precipitation"]))
        _weather_cache["records"] = records
        _weather_cache["fetched_at"] = t_now
        _weather_retry_after = None
        return records
    except Exception as e:
        _weather_retry_after = t_now + timedelta(minutes=5)
        print(f"[weather] fetch failed: {e} -- reusing last cache, retry in 5 min")
        return _weather_cache["records"]


def fetch_seismic(t_start: datetime, t_end: datetime):
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&orderby=time"
           f"&starttime={t_start.strftime('%Y-%m-%dT%H:%M:%S')}"
           f"&endtime={t_end.strftime('%Y-%m-%dT%H:%M:%S')}"
           f"&latitude={LAKE_LAT}&longitude={LAKE_LON}&maxradiuskm=300&minmagnitude=2.0")
    events = []
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        for f in r.json().get("features", []):
            eid = f.get("id")
            props = f["properties"]
            if eid in _seismic_seen_ids:
                continue
            _seismic_seen_ids.add(eid)
            events.append({"time": datetime.fromtimestamp(props["time"] / 1000, tz=timezone.utc),
                           "mag": props.get("mag") or 0.0})
    except Exception as e:
        print(f"[seismic] fetch failed: {e} -- assuming 0 events")
    return events


def latest_hourly_value(series, t_now, index):
    valid = [s for s in series if s[0] <= t_now]
    if not valid:
        return None
    return max(valid, key=lambda x: x[0])[index]


def get_water_level(t_now, temp, precip, seismic_energy, history):
    """Synthetic, mean-reverting water level driven by REAL rain / temperature / seismic inputs.
    Swap for a real gauge fetch if one becomes available."""
    prev = history[-1]["water_level_m"] if history else WATER_BASE_M
    target = (WATER_BASE_M
              + WATER_RAIN_GAIN_M_PER_MM * (precip or 0.0)
              + WATER_MELT_GAIN_M_PER_C * max(0.0, temp or 0.0))
    seismic_effect = min(seismic_energy, 1e6) * 1e-8
    noise = random.uniform(-0.005, 0.005)
    new_level = prev + WATER_RELAX * (target - prev) + seismic_effect + noise
    return round(new_level, 4), "synthetic"


def run_step(t_now: datetime):
    global _sim_clock, _last_seismic_slot
    t_real = round_to_10min(t_now)

    if SIM_STEP_MIN:
        _sim_clock = t_real if _sim_clock is None else _sim_clock + timedelta(minutes=SIM_STEP_MIN)
        t_row = _sim_clock
    else:
        t_row = t_real

    weather = fetch_weather(t_real)
    if t_real != _last_seismic_slot:
        seismic_events = fetch_seismic(t_real - timedelta(minutes=SEISMIC_WINDOW_MIN), t_real)
        _last_seismic_slot = t_real
    else:
        seismic_events = []

    temperature_c = latest_hourly_value(weather, t_real, 1)
    precipitation_mm = latest_hourly_value(weather, t_real, 2)
    seismic_energy = sum(10 ** (1.5 * e["mag"]) for e in seismic_events)

    water_level_m, water_status = get_water_level(t_row, temperature_c, precipitation_mm,
                                                  seismic_energy, _history)
    row = {
        "timestamp_utc": t_row.isoformat(),
        "water_level_m": water_level_m,
        "vibration_rms_g": round(random.uniform(0.02, 0.05), 5),
        "tilt_deg": round(random.uniform(-0.01, 0.01), 5),
        "rainfall_mm_hr": precipitation_mm if precipitation_mm is not None else 0.0,
        "turbidity_ntu": round(random.uniform(2.5, 6.0), 3),
        "ambient_temp_c": temperature_c if temperature_c is not None else 0.0,
        "seismic_energy": round(seismic_energy, 2),
        "water_valid": 1,
        "vibration_valid": 1,
        "tilt_valid": 1,
        "rainfall_valid": 1 if precipitation_mm is not None else 0,
        "turbidity_valid": 1,
        "seismic_valid": 1,
        "sensor_status_water": water_status,
        "is_synthetic_water": True,
    }
    _history.append(row)
    return row


def append_to_csv(row: dict):
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    print(f"GlacierGuard real-time feed starting. Polling every {POLL_INTERVAL_SEC}s.")
    print(f"Lake coords: ({LAKE_LAT}, {LAKE_LON}) | Output: {OUTPUT_CSV}")
    try:
        while True:
            row = run_step(datetime.now(timezone.utc))
            append_to_csv(row)
            print(f"[{row['timestamp_utc']}] water={row['water_level_m']}m  rain={row['rainfall_mm_hr']}mm/hr  "
                  f"temp={row['ambient_temp_c']}C  seismic_energy={row['seismic_energy']}  "
                  f"status={row['sensor_status_water']}")
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()