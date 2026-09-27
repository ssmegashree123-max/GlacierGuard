"""
satellite_risk.py  --  GlacierGuard satellite risk layer (uses YOUR Sentinel-2 dataset)
========================================================================================
Reads satellite_data/scenes.csv (built by build_satellite_dataset.py) and produces:

  1. A site-level satellite context: how full is the lake vs its pre-breach level,
     is it refilling, is water pressing on the basin edge, how old is the latest scene.
  2. An "event detector": a sudden collapse of lake area between two consecutive QC-good
     scenes (this is exactly what the real 3-4 Oct 2023 breach looks like in your data).
  3. A probability adjuster: p_adjusted = odds(p_model) * multiplier  (multiplier in [1, 1.5],
     or 3.0 when a breach event is active).
  4. satellite_status.json for the dashboard.

WHY THIS IS NOT A TRAINING FEATURE (say this in the demo)
  The telemetry model is trained on synthetic trajectories that have no real calendar dates,
  so they cannot be aligned with real satellite scenes. A satellite number that is identical for
  every training row carries zero information for XGBoost. Instead the satellite data acts as a
  transparent, rule-based prior applied AFTER the model. The weights below are hand-set
  heuristics, not learned from data.

Usage
  python satellite_risk.py                      # context as of today, writes satellite_status.json
  python satellite_risk.py --as-of 2023-10-10   # replay: what did the satellite layer say then?
  python satellite_risk.py --backtest           # replay every good scene, list detected events
  python satellite_risk.py --selftest           # quick checks against your real scenes.csv
From run_demo.py:
  import satellite_risk
  sat = satellite_risk.get_satellite_context()
  p_gradual_adj = satellite_risk.adjust_probability(p_gradual, sat)
"""
import glob
import json
import math
import os
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "satellite_status.json")

# ------------------------- heuristic settings (hand-set) --------------------
EVENT_LOWER_DROP_FRAC = 0.30     # lower-bound water area falls >=30% between consecutive good scenes
EVENT_ACTIVE_DAYS = 45           # how long a detected event keeps the alert active
EVENT_MULTIPLIER = 3.0
MAX_PRIOR_MULTIPLIER = 1.5       # cap for the non-event prior
FILL_START, FILL_FULL = 0.85, 1.00   # fill fraction range mapped to score 0..1
TREND_FULL_KM2_PER_YR = 0.10     # growth rate treated as "full" trend score
EDGE_START_PCT, EDGE_FULL_PCT = 0.5, 2.0   # % of upper-bound water in the basin's inner edge band
FRESHNESS_TAU_DAYS = 45.0        # satellite evidence decays with scene age
WEIGHTS = {"fill": 0.5, "trend": 0.3, "edge": 0.2}
LEVEL_ELEVATED, LEVEL_WATCH = 1.30, 1.10


def find_scenes_csv():
    for root in (os.path.join(HERE, "satellite_data"), HERE, "."):
        p = os.path.join(root, "scenes.csv")
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(HERE, "**", "scenes.csv"), recursive=True)
    return hits[0] if hits else None


def load_scenes(path=None):
    path = path or find_scenes_csv()
    if not path:
        raise FileNotFoundError("scenes.csv not found (expected in ./satellite_data/)")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True), os.path.dirname(path)


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


# ------------------------------ event detection -----------------------------
def detect_events(scenes):
    """Replay consecutive QC-good scenes; return list of sudden lake-area collapses."""
    good = scenes[scenes["qc"] == "good"].reset_index(drop=True)
    events = []
    for i in range(1, len(good)):
        prev, cur = good.iloc[i - 1], good.iloc[i]
        if prev["water_lower_km2"] <= 0:
            continue
        drop = (prev["water_lower_km2"] - cur["water_lower_km2"]) / prev["water_lower_km2"]
        if drop >= EVENT_LOWER_DROP_FRAC:
            events.append({
                "detected_on": cur["date"].strftime("%Y-%m-%d"),
                "previous_scene": prev["date"].strftime("%Y-%m-%d"),
                "lower_km2_before": float(prev["water_lower_km2"]),
                "lower_km2_after": float(cur["water_lower_km2"]),
                "lower_drop_pct": round(drop * 100, 1),
            })
    return events


# ------------------------------ main context --------------------------------
def compute_context(scenes, as_of=None):
    as_of = pd.Timestamp(as_of or datetime.now(timezone.utc).date())
    scenes = scenes[scenes["date"] <= as_of]
    good = scenes[scenes["qc"] == "good"]
    if good.empty:
        return {"available": False, "level": "UNKNOWN", "multiplier": 1.0,
                "message": "No usable satellite scenes before this date", "alert_active": False}

    pre = good[good["phase"] == "pre_breach"]
    ref_src = pre if not pre.empty else good
    ref_upper = float(ref_src["water_upper_km2"].max())
    ref_lower = float(ref_src["water_lower_km2"].max())

    recent = good.tail(3)
    upper_now = float(recent["water_upper_km2"].median())
    lower_now = float(recent["water_lower_km2"].median())
    fill = upper_now / ref_upper

    # trend: slope of yearly median upper area over the last 3 distinct years (km2 / yr)
    yearly = good.groupby(good["date"].dt.year)["water_upper_km2"].median()
    last_years = yearly.tail(3)
    growth = float(np.polyfit(last_years.index.values, last_years.values, 1)[0]) if len(last_years) >= 2 else 0.0

    edge_pct = float(recent["edge_pct_of_upper"].mean())

    latest = good.iloc[-1]
    age_days = int((as_of - latest["date"]).days)
    freshness = math.exp(-max(age_days, 0) / FRESHNESS_TAU_DAYS)

    fill_s = _clip01((fill - FILL_START) / (FILL_FULL - FILL_START))
    trend_s = _clip01(growth / TREND_FULL_KM2_PER_YR)
    edge_s = _clip01((edge_pct - EDGE_START_PCT) / (EDGE_FULL_PCT - EDGE_START_PCT))
    raw_boost = WEIGHTS["fill"] * fill_s + WEIGHTS["trend"] * trend_s + WEIGHTS["edge"] * edge_s
    multiplier = min(MAX_PRIOR_MULTIPLIER, 1.0 + raw_boost * freshness)

    events = detect_events(scenes)
    active = None
    for ev in events:
        d = (as_of - pd.Timestamp(ev["detected_on"])).days
        if 0 <= d <= EVENT_ACTIVE_DAYS:
            active = dict(ev, days_since=int(d))
    if active:
        multiplier, level = EVENT_MULTIPLIER, "ALERT"
        message = (f"Satellite detected a sudden lake-area collapse ({active['lower_drop_pct']}% drop, "
                   f"{active['previous_scene']} -> {active['detected_on']}): probable outburst")
    elif multiplier >= LEVEL_ELEVATED:
        level, message = "ELEVATED", f"Lake at {fill*100:.0f}% of pre-breach extent and still growing"
    elif multiplier >= LEVEL_WATCH:
        level, message = "WATCH", (f"Lake refilling: {fill*100:.0f}% of pre-breach extent, "
                                   f"+{growth:.3f} km2/yr")
    else:
        level, message = "NORMAL", "No satellite-based concern"

    tail_warn = []
    if age_days > 30:
        tail_warn.append(f"latest good scene is {age_days} days old")
    if edge_pct > 1.0:
        tail_warn.append(f"{edge_pct:.2f}% of water touches the basin edge band (possible growth beyond the 2022 outline)")

    series = good[["date", "phase", "water_lower_km2", "water_upper_km2"]].copy()
    series["date"] = series["date"].dt.strftime("%Y-%m-%d")

    return {
        "available": True,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "level": level,
        "multiplier": round(float(multiplier), 3),
        "alert_active": bool(active),
        "message": message,
        "warnings": tail_warn,
        "latest_good_scene": latest["date"].strftime("%Y-%m-%d"),
        "scene_age_days": age_days,
        "freshness": round(freshness, 3),
        "lake_upper_km2": round(upper_now, 4),
        "lake_lower_km2": round(lower_now, 4),
        "pre_breach_ref_upper_km2": round(ref_upper, 4),
        "pre_breach_ref_lower_km2": round(ref_lower, 4),
        "fill_fraction": round(fill, 4),
        "growth_km2_per_yr": round(growth, 4),
        "edge_pct_recent": round(edge_pct, 3),
        "scores": {"fill": round(fill_s, 3), "trend": round(trend_s, 3), "edge": round(edge_s, 3)},
        "active_event": active,
        "all_events_on_record": events,
        "series": series.to_dict(orient="list"),
        "latest_images": {"rgb": latest.get("png_rgb"), "overlay": latest.get("png_overlay")},
    }


def _breach_replay_images(scenes):
    """Before/after image pair around the first detected event, for the demo visual."""
    events = detect_events(scenes)
    if not events:
        return None
    ev = events[0]
    idx = scenes.set_index(scenes["date"].dt.strftime("%Y-%m-%d"))
    out = {}
    for tag, d in (("before", ev["previous_scene"]), ("after", ev["detected_on"])):
        if d in idx.index:
            row = idx.loc[d]
            out[tag] = {"date": d, "rgb": row["png_rgb"], "overlay": row["png_overlay"]}
    return out or None


def get_satellite_context(as_of=None, scenes_csv=None, write=True):
    """Safe entry point: never raises; returns a dict (level UNKNOWN on failure)."""
    try:
        scenes, base = load_scenes(scenes_csv)
        ctx = compute_context(scenes, as_of)
        ctx["images_dir"] = os.path.relpath(os.path.join(base, "images"), HERE).replace("\\", "/")
        ctx["breach_replay_images"] = _breach_replay_images(scenes)
    except Exception as e:
        # If the raw scenes file is not present, keep a previously generated
        # satellite context visible rather than pretending satellite is live.
        # The dashboard will show that this is cached context.
        cached = None
        if os.path.exists(OUT_JSON):
            try:
                with open(OUT_JSON, "r", encoding="utf-8") as f:
                    cached = json.load(f)
            except Exception:
                cached = None
        if isinstance(cached, dict) and cached.get("level") not in (None, "UNKNOWN"):
            cached.setdefault("warnings", [])
            cached["warnings"] = list(cached["warnings"]) + [
                "Using cached satellite context because scenes.csv is not available in this run."
            ]
            cached["satellite_data_mode"] = "cached"
            ctx = cached
        else:
            ctx = {"available": False, "level": "UNKNOWN", "multiplier": 1.0, "alert_active": False,
                   "message": f"Satellite layer unavailable: {e}"}
    if write:
        try:
            tmp = OUT_JSON + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(ctx, f, indent=2)
            os.replace(tmp, OUT_JSON)
        except Exception as e:
            print(f"[satellite] could not write {OUT_JSON}: {e}")
    return ctx


def adjust_probability(p, ctx):
    """Scale the model's odds by the satellite multiplier. Returns a probability in [0, 1]."""
    m = float((ctx or {}).get("multiplier", 1.0))
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    odds = p / (1 - p) * m
    return odds / (1 + odds)


# ---------------------------------- CLI -------------------------------------
def _backtest():
    scenes, _ = load_scenes()
    print("Replaying QC-good scenes in date order (event = lower-bound area drop "
          f">= {EVENT_LOWER_DROP_FRAC*100:.0f}% between consecutive good scenes)\n")
    events = detect_events(scenes)
    if not events:
        print("No events detected.")
    for ev in events:
        print(f"EVENT detected {ev['detected_on']}: lower {ev['lower_km2_before']:.3f} -> "
              f"{ev['lower_km2_after']:.3f} km2 ({ev['lower_drop_pct']}% drop) vs {ev['previous_scene']}")
    good = scenes[scenes["qc"] == "good"].reset_index(drop=True)
    drops = []
    for i in range(1, len(good)):
        p, c = good.iloc[i - 1], good.iloc[i]
        drops.append((p["water_lower_km2"] - c["water_lower_km2"]) / p["water_lower_km2"])
    print(f"\nLargest 'drop' between consecutive good scenes EXCLUDING detected events: "
          f"{max([d for d in drops if d < EVENT_LOWER_DROP_FRAC] or [0])*100:.1f}%  (noise floor)")
    for d in ("2023-09-26", "2023-10-10", "2023-11-10", "2023-12-31"):
        c = compute_context(scenes, d)
        print(f"as of {d}: level={c['level']:<9} multiplier={c['multiplier']}  {c['message']}")


def _selftest():
    scenes, _ = load_scenes()
    ev = detect_events(scenes)
    assert len(ev) == 1 and ev[0]["detected_on"] == "2023-10-09", ev
    c_before = compute_context(scenes, "2023-09-30")
    c_after = compute_context(scenes, "2023-10-10")
    assert not c_before["alert_active"] and c_after["alert_active"] and c_after["multiplier"] == EVENT_MULTIPLIER
    c_now = compute_context(scenes, "2026-09-24")
    assert 1.0 <= c_now["multiplier"] <= MAX_PRIOR_MULTIPLIER and not c_now["alert_active"], c_now
    assert 0.0 < adjust_probability(0.5, c_now) < 1.0 and abs(adjust_probability(0.5, {"multiplier": 1}) - 0.5) < 1e-9
    assert adjust_probability(0.3, c_now) > 0.3
    ctx = get_satellite_context(as_of="2026-09-24", write=True)
    assert os.path.exists(OUT_JSON) and json.load(open(OUT_JSON))["level"] == ctx["level"]
    empty = compute_context(scenes, "2010-01-01")
    assert empty["level"] == "UNKNOWN"
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--backtest" in sys.argv:
        _backtest()
    else:
        as_of = None
        if "--as-of" in sys.argv:
            as_of = sys.argv[sys.argv.index("--as-of") + 1]
        out = get_satellite_context(as_of=as_of)
        out.pop("series", None)
        print(json.dumps(out, indent=2))
        print(f"\nWrote {OUT_JSON}")