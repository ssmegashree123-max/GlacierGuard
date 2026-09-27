"""
GlacierGuard v3 - Live inference on live_telemetry.csv
========================================================
Loads the model trained by train_model_v3.py and scores the live sensor
feed (produced by realtime_feed.py) with it, using the SAME causal feature
logic as training -- so features seen at inference match training exactly.

The live feed is treated as one continuous trajectory ("LIVE"). Every row
appended to live_telemetry.csv is re-scored using the full history seen so
far (rolling windows need trailing context), and we print the current
escalation probability + a smoothed alert state.

Run after train_model_v3.py has produced glacierguard_v3_outputs/:
    python live_inference.py
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from datetime import datetime

from train_model_v3 import build_features, RAW_SENSORS, VALID_PAIRING
from risk_engine import classify_risk  # reuse exact same logic

MODEL_DIR = "glacierguard_v3_outputs"
LIVE_CSV = "live_telemetry.csv"
ALERT_SMOOTH_ROWS = 10       # 10 rows trailing mean, matches notebook's alert smoothing
ALERT_THRESHOLD = 0.5


def load_models():
    with open(f"{MODEL_DIR}/features.json") as fh:
        features = json.load(fh)
    sat_path = f"{MODEL_DIR}/satellite_site_features.json"
    satellite_features = None
    if os.path.exists(sat_path):
        with open(sat_path) as fh:
            satellite_features = json.load(fh)

    clf_gradual = XGBClassifier()
    clf_gradual.load_model(f"{MODEL_DIR}/glacierguard_gradual.json")
    clf_sudden = XGBClassifier()
    clf_sudden.load_model(f"{MODEL_DIR}/glacierguard_sudden.json")
    reg_ttc = XGBRegressor()
    reg_ttc.load_model(f"{MODEL_DIR}/glacierguard_ttc.json")

    return features, satellite_features, clf_gradual, clf_sudden, reg_ttc


def prepare_live_df(live_csv):
    """
    Adapt live_telemetry.csv's schema to what build_features() expects:
    - synthesize trajectory_id = "LIVE" (single continuous stream)
    - convert timestamp_utc -> timestamp_sec (seconds since first reading)
    - map validity columns directly (schemas already align on names)
    """
    df = pd.read_csv(live_csv)
    if df.empty:
        return df

    df["trajectory_id"] = "LIVE"
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="ISO8601")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    t0 = df["timestamp_utc"].iloc[0]
    df["timestamp_sec"] = (df["timestamp_utc"] - t0).dt.total_seconds()

    # historical training data has a `system_state` column the live feed doesn't --
    # build_features() only uses it if present, so it's fine to omit it here.
    return df


def score_live_feed():
    features, satellite_features, clf_gradual, clf_sudden, reg_ttc = load_models()

    live_df = prepare_live_df(LIVE_CSV)
    if live_df.empty or len(live_df) < 2:
        print("Not enough live rows yet to compute rolling features "
              "(need at least 2). Let realtime_feed.py collect more data.")
        return

    feat = build_features(live_df, satellite_features)
    # align to the exact training feature columns -- any missing (e.g. a
    # system_state one-hot the live feed never triggered) becomes 0
    X_live = feat.reindex(columns=features, fill_value=0.0)

    p_gradual = clf_gradual.predict_proba(X_live)[:, 1]
    p_sudden = clf_sudden.predict_proba(X_live)[:, 1]
    ttc = reg_ttc.predict(X_live)

    feat["p_gradual"] = p_gradual
    feat["p_sudden"] = p_sudden
    feat["p_gradual_smooth"] = feat["p_gradual"].rolling(
        ALERT_SMOOTH_ROWS, min_periods=1).mean()
    feat["p_sudden_smooth"] = feat["p_sudden"].rolling(
        ALERT_SMOOTH_ROWS, min_periods=1).mean()
    feat["time_to_critical_min"] = ttc

    latest = feat.iloc[-1]
    latest_ts = live_df["timestamp_utc"].iloc[-1]

    gradual_alert = latest["p_gradual_smooth"] >= ALERT_THRESHOLD
    sudden_alert = latest["p_sudden_smooth"] >= ALERT_THRESHOLD

    risk = classify_risk(
        latest["p_gradual_smooth"],
        latest["p_sudden_smooth"],
        latest["time_to_critical_min"]
    )

    print(f"=== GlacierGuard live risk assessment @ {latest_ts} ===")
    print(f"  Rows of live history used: {len(live_df)}")
    print(f"  Gradual escalation (2h) probability : {latest['p_gradual']:.3f} "
          f"(smoothed: {latest['p_gradual_smooth']:.3f}) "
          f"-> {'ALERT' if gradual_alert else 'normal'}")
    print(f"  Sudden impact probability            : {latest['p_sudden']:.3f} "
          f"(smoothed: {latest['p_sudden_smooth']:.3f}) "
          f"-> {'ALERT' if sudden_alert else 'normal'}")

    print(f"  Time to critical                     : {latest['time_to_critical_min']:.1f} min")
    print(f"  Risk level                           : {risk['risk_level']}")
    print(f"  Decision                             : {risk['decision']}")

    if gradual_alert or sudden_alert:
        print("\n  RECOMMENDED ACTION:")
        print("  -> Notify emergency coordination center")
        print("  -> Cross-check with downstream cascade module")
        print("  -> Consider issuing downstream warning")

    feat.attrs["risk_result"] = risk
    return feat


if __name__ == "__main__":
    score_live_feed()