"""
GlacierGuard - One-command demo runner (v3)
=============================================
  python run_demo.py                      # live mode: real weather/seismic, accelerated 10-min sim clock
  python run_demo.py --scenario           # replay an unseen escalating test trajectory -> ALERT on camera
  python run_demo.py --scenario --sat-as-of 2023-10-10    # also show the satellite layer's real 2023 breach ALERT
  python run_demo.py --tick 20            # seconds between ticks (default 60 live, 3 scenario)

Every start is a FRESH session: the old live_telemetry.csv is deleted, because leftover rows from an
earlier run (different water-level history, timestamp gaps) corrupt the rolling features.

Then open:
    http://localhost:8000/dashboard.html
    http://localhost:8000/satellite_panel.html
Stop with Ctrl+C.
"""

import argparse
import json
import math
import os
import threading
import http.server
import socketserver
import webbrowser
from datetime import datetime, timezone

import realtime_feed as feed
import live_inference as infer
import satellite_risk

LIVE_TICK_SEC = 5
SCENARIO_TICK_SEC = 3
SIM_STEP_MIN = 10            # each live row = 10 virtual minutes (the model's training cadence)
HTTP_PORT = 8000
STATUS_FILE = "status.json"

SATELLITE_AS_OF = None       # or "2023-10-10" (also settable with --sat-as-of)

# False (default): the gradual ALERT uses the model's own probability OR a satellite breach event.
# The satellite-adjusted probability is still computed and shown.
SATELLITE_ADJUSTS_ALERT = False

_replay = None               # ScenarioReplay instance in --scenario mode


def _clean(v):
    """JSON cannot hold NaN/inf (dashboard fetch().json() would fail) -> None."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def compact_satellite(sat):
    keys = ["available", "level", "multiplier", "alert_active", "message", "warnings", "as_of",
            "latest_good_scene", "scene_age_days", "fill_fraction", "lake_lower_km2",
            "lake_upper_km2", "growth_km2_per_yr", "edge_pct_recent"]
    return {k: sat.get(k) for k in keys if k in sat}


def write_status(row, score_result, sat=None, mode="live"):
    status = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "latest_reading": {k: _clean(row[k]) for k in (
            "water_level_m", "rainfall_mm_hr", "ambient_temp_c", "seismic_energy",
            "vibration_rms_g", "tilt_deg", "turbidity_ntu", "sensor_status_water")},
        "risk": score_result,
        "satellite": compact_satellite(sat) if sat else None,
        "rows_of_history": score_result.get("rows_of_history", 0) if score_result else 0,
    }
    status["latest_reading"]["timestamp"] = row["timestamp_utc"]
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, STATUS_FILE)


def start_fresh_session():
    for f in (feed.OUTPUT_CSV, STATUS_FILE):
        if os.path.exists(f):
            os.remove(f)
    feed.reset_state()
    if _replay is not None:
        for r in _replay.warmup_rows():
            feed.append_to_csv(r)
        print(f"[scenario] {_replay.describe()}")
        print(f"[scenario] wrote {_replay.warmup_end} warm-up rows; escalation begins ~30 ticks into the replay")
    else:
        feed.SIM_STEP_MIN = SIM_STEP_MIN
        print(f"[demo] prototype live mode: live weather/seismic + simulated sensor channels, {SIM_STEP_MIN}-minute virtual row spacing")


def run_one_tick(t_now=None):
    if _replay is not None:
        row = _replay.next_row()
        if row is None:
            print("[scenario] replay finished")
            return None, None, None
        feed.append_to_csv(row)
    else:
        row = feed.run_step(t_now or datetime.now(timezone.utc))
        feed.append_to_csv(row)
    print(f"[feed] {row['timestamp_utc']}  water={row['water_level_m']:.3f}m  "
          f"rain={row['rainfall_mm_hr']}mm/hr  temp={row['ambient_temp_c']}C  seismic={row['seismic_energy']}")

    sat = satellite_risk.get_satellite_context(as_of=SATELLITE_AS_OF)

    score_result = None
    try:
        feat = infer.score_live_feed()
        if feat is not None and len(feat) > 0:
            latest = feat.iloc[-1]
            p_grad_smooth = float(latest["p_gradual_smooth"])
            p_grad_adj = satellite_risk.adjust_probability(p_grad_smooth, sat)
            from risk_engine import classify_risk
            final_gradual = p_grad_adj if SATELLITE_ADJUSTS_ALERT else p_grad_smooth
            risk = classify_risk(
                final_gradual,
                float(latest["p_sudden_smooth"]),
                float(latest.get("time_to_critical_min", 0)),
                satellite_alert=bool(sat.get("alert_active", False))
            )
            score_result = {
                "p_gradual": round(float(latest["p_gradual"]), 4),
                "p_gradual_smooth": round(p_grad_smooth, 4),
                "p_gradual_satellite_adjusted": round(p_grad_adj, 4),
                "p_sudden": round(float(latest["p_sudden"]), 4),
                "p_sudden_smooth": round(float(latest["p_sudden_smooth"]), 4),
                "time_to_critical_min": round(float(latest.get("time_to_critical_min", 0)), 1),
                "risk_level": risk["risk_level"],
                "lead_time_min": risk["lead_time_min"],
                "alert": risk["alert"],
                "decision": risk["decision"],
                "gradual_alert": bool(final_gradual >= infer.ALERT_THRESHOLD),
                "sudden_alert": bool(latest["p_sudden_smooth"] >= infer.ALERT_THRESHOLD),
                "satellite_event_alert": bool(sat.get("alert_active", False)),
                "rows_of_history": len(feat),
            }
            print(f"[risk] gradual model={p_grad_smooth:.3f} -> satellite-adjusted={p_grad_adj:.3f} "
                  f"(x{sat.get('multiplier', 1.0)}, {sat.get('level')})")
    except Exception as e:
        print(f"[inference] not enough history yet or error: {e}")

    write_status(row, score_result, sat, mode="scenario" if _replay is not None else "live")
    return row, score_result, sat


def run_pipeline_loop(tick_sec):
    print(f"[demo] Pipeline loop starting. Tick every {tick_sec}s.")
    while True:
        run_one_tick()
        threading.Event().wait(tick_sec)


def start_http_server():
    socketserver.TCPServer.allow_reuse_address = True
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", HTTP_PORT), handler) as httpd:
        print(f"[demo] Dashboard server running at http://localhost:{HTTP_PORT}/dashboard.html")
        httpd.serve_forever()


def main():
    global _replay, SATELLITE_AS_OF
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", action="store_true", help="replay an unseen escalating test trajectory")
    ap.add_argument("--sat-as-of", default=None, help="satellite layer replay date, e.g. 2023-10-10")
    ap.add_argument("--tick", type=float, default=None, help="seconds between ticks")
    args = ap.parse_args()
    if args.sat_as_of:
        SATELLITE_AS_OF = args.sat_as_of
    if args.scenario:
        from scenario_replay import ScenarioReplay
        _replay = ScenarioReplay()
    tick = args.tick if args.tick is not None else (SCENARIO_TICK_SEC if args.scenario else LIVE_TICK_SEC)
    feed.POLL_INTERVAL_SEC = tick

    start_fresh_session()
    threading.Thread(target=start_http_server, daemon=True).start()
    try:
        webbrowser.open(f"http://localhost:{HTTP_PORT}/dashboard.html")
    except Exception:
        pass
    try:
        run_pipeline_loop(tick)
    except KeyboardInterrupt:
        print("\n[demo] Stopped.")


if __name__ == "__main__":
    main()