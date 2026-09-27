"""
GlacierGuard v3 - Training pipeline updated for the live seismic channel
==========================================================================
This extends your existing notebook's approach (causal rolling features,
grouped-by-trajectory split, XGBoost with early stopping, satellite prior)
to add `seismic_energy` (from the live USGS feed) as a 7th sensor channel,
so the trained model's feature schema matches what `live_telemetry.csv`
actually produces.

IMPORTANT, read before you present this:
Your historical synthetic_glacier_telemetry.csv has NO real seismic
readings (it predates the live feed). We backfill seismic_energy = 0.0 for
every historical row so the schema lines up. This means the model CANNOT
yet learn "seismic activity precedes escalation" -- there's no variation in
the training signal to learn from. Adding the column now is about making
the *pipeline* future-proof (same input shape as live inference), not about
seismic actually improving accuracy today. Be upfront about this in your
demo: "seismic is wired into the pipeline; it will start contributing once
we have training trajectories with real seismic variation."

Everything else mirrors your existing notebook 1:1 -- same causal
forward-fill logic, same SHORT/LONG rolling windows, same satellite prior,
same grouped TRAIN/VAL/TEST_UNSEEN split, same XGBoost hyperparameters and
early stopping, same alert-level (lead time / false alarms per day) eval.
"""

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_curve, confusion_matrix, accuracy_score,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Config -- 7 sensors now (added seismic_energy), matching the live feed
# ---------------------------------------------------------------------------

RAW_SENSORS = [
    "water_level_m", "vibration_rms_g", "tilt_deg",
    "rainfall_mm_hr", "turbidity_ntu", "ambient_temp_c",
    "seismic_energy",                                    # NEW
]
VALID_COLS = [
    "water_valid", "vibration_valid", "tilt_valid",
    "rainfall_valid", "turbidity_valid", "seismic_valid",  # NEW
]
# ambient_temp has no own validity flag in the historical schema -> reuse
# water_valid for it (unchanged from your notebook), seismic gets its own.
VALID_PAIRING = [
    "water_valid", "vibration_valid", "tilt_valid",
    "rainfall_valid", "turbidity_valid", "water_valid", "seismic_valid",
]

# Satellite handling (see satellite_risk.py):
# The synthetic training trajectories have no real dates, so site-level satellite numbers are
# IDENTICAL on every training row. A constant column gives XGBoost zero information, so by default
# they are NOT added to the model features. The satellite data is applied AFTER the model as a
# rule-based prior by satellite_risk.py. Set True only to reproduce the old v3 behaviour.
SATELLITE_AS_MODEL_FEATURES = False

# `system_state` is a column of the SIMULATOR (ground-truth state of each synthetic trajectory).
# A real deployment has no such sensor, and live_telemetry.csv does not contain it, so live
# inference feeds zeros for every state_* column. A model that relies on them (a) sees a different
# input at inference than in training and (b) may have inflated validation/test scores if the state
# encodes the escalation phase. Default False = train only on features the live feed can supply.
# Retrain once with True and once with False and compare PR-AUC to see how much state was helping.
USE_SYSTEM_STATE_FEATURES = False

CLF_TARGETS = ["target_gradual_escalates_2hr", "target_sudden_impact"]
REG_TARGET = "time_to_critical_min"
SHORT_WIN, LONG_WIN = 5, 20
OUT_DIR = "glacierguard_v3_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def find_input_file(filename):
    candidate_dirs = [
        ".", "/mnt/user-data/uploads", "/mnt/data", "/content",
        "/kaggle/input", os.path.expanduser("~"),
    ]
    for d in candidate_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    for root in (".", "/kaggle/input", "/mnt/user-data/uploads"):
        if os.path.isdir(root):
            matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
            if matches:
                return matches[0]
    return None


# ---------------------------------------------------------------------------
# Step 1: load + backfill seismic_energy on historical data
# ---------------------------------------------------------------------------

def load_and_prepare_raw(telemetry_csv):
    raw = pd.read_csv(telemetry_csv)

    # Remove only malformed rows that cannot belong to a trajectory/split.
    # This fixes the old blank row that made TRAJ_001 appear in two splits.
    required = ["trajectory_id", "timestamp_sec", "split"]
    before = len(raw)
    raw = raw.dropna(subset=required).copy()
    raw["trajectory_id"] = raw["trajectory_id"].astype(str).str.strip()
    raw["split"] = raw["split"].astype(str).str.strip()
    raw = raw[(raw["trajectory_id"] != "") & (raw["split"] != "")].copy()
    dropped = before - len(raw)
    if dropped:
        print(f"Cleaned {dropped} malformed row(s) with missing trajectory_id/timestamp/split.")

    raw = raw.sort_values(["trajectory_id", "timestamp_sec"]).reset_index(drop=True)

    if "seismic_energy" not in raw.columns:
        print("NOTE: historical data has no seismic_energy column -- "
              "backfilling 0.0 (no real seismic signal in training data yet).")
        raw["seismic_energy"] = 0.0
    if "seismic_valid" not in raw.columns:
        raw["seismic_valid"] = 1

    mixed = raw.groupby("trajectory_id")["split"].nunique()
    bad = mixed[mixed != 1]
    if not bad.empty:
        raise ValueError(
            "Trajectory leakage detected: these trajectory_ids occur in multiple splits: "
            + ", ".join(bad.index.astype(str))
        )
    expected_splits = {"TRAIN", "VAL", "TEST_UNSEEN"}
    missing = expected_splits - set(raw["split"].unique())
    if missing:
        raise ValueError(f"Missing required split(s): {sorted(missing)}")
    return raw


# ---------------------------------------------------------------------------
# Step 2: satellite prior (unchanged from your notebook)
# ---------------------------------------------------------------------------

def build_satellite_features(yearly_csv, scenes_csv):
    """Site-level satellite summary from the real Sentinel-2 dataset (scenes.csv).
    Used for reporting / satellite_site_features.json; only fed to the model if
    SATELLITE_AS_MODEL_FEATURES is True."""
    scenes = pd.read_csv(scenes_csv)
    scenes["date"] = pd.to_datetime(scenes["date"])
    good = scenes[scenes["qc"] == "good"].sort_values("date")
    pre = good[good["phase"] == "pre_breach"]
    ref = pre if not pre.empty else good

    pre_year = ref.groupby(ref["date"].dt.year)["water_upper_km2"].median()
    growth_km2_per_yr = (float(np.polyfit(pre_year.index.values, pre_year.values, 1)[0])
                         if len(pre_year) >= 2 else 0.0)

    return {
        "site_lake_area_km2_latest": float(good["water_upper_km2"].iloc[-1]),
        "site_lake_area_km2_mean": float(good["water_upper_km2"].mean()),
        "site_lake_growth_km2_per_yr": growth_km2_per_yr,          # pre-breach growth
        "site_scene_good_quality_frac": float((scenes["qc"] == "good").mean()),
        "site_had_breach_on_record": float((scenes["phase"] != "pre_breach").any()),
        "site_pre_breach_max_upper_km2": float(ref["water_upper_km2"].max()),
    }


# ---------------------------------------------------------------------------
# Step 3: causal feature engineering -- same logic as your notebook,
# now looping over 7 sensors instead of 6
# ---------------------------------------------------------------------------

def build_features(df, satellite_features=None):
    df = df.sort_values(["trajectory_id", "timestamp_sec"]).copy()
    pairs = list(zip(RAW_SENSORS, VALID_PAIRING))

    out_cols = {}
    for sensor, vcol in pairs:
        raw_s = df[sensor]
        valid = df[vcol] == 1
        masked = raw_s.where(valid)
        filled = masked.groupby(df["trajectory_id"]).ffill()
        traj_median = filled.groupby(df["trajectory_id"]).transform("median")
        filled = filled.fillna(traj_median)

        out_cols[f"{sensor}_f"] = filled
        streak_id = valid.groupby(df["trajectory_id"]).cumsum()
        out_cols[f"{sensor}_stale_count"] = (
            (~valid).astype(int).groupby([df["trajectory_id"], streak_id]).cumsum()
        )

    feat = pd.DataFrame(out_cols, index=df.index)

    for sensor, _ in pairs:
        s = feat[f"{sensor}_f"]
        gs = s.groupby(df["trajectory_id"])
        feat[f"{sensor}_roll_mean_{SHORT_WIN}"] = gs.transform(
            lambda x: x.rolling(SHORT_WIN, min_periods=1).mean())
        feat[f"{sensor}_roll_std_{SHORT_WIN}"] = gs.transform(
            lambda x: x.rolling(SHORT_WIN, min_periods=1).std()).fillna(0.0)
        feat[f"{sensor}_roll_mean_{LONG_WIN}"] = gs.transform(
            lambda x: x.rolling(LONG_WIN, min_periods=1).mean())
        feat[f"{sensor}_roll_std_{LONG_WIN}"] = gs.transform(
            lambda x: x.rolling(LONG_WIN, min_periods=1).std()).fillna(0.0)
        dt = df.groupby("trajectory_id")["timestamp_sec"].diff().replace(0, np.nan)
        diff = gs.transform(lambda x: x.diff())
        feat[f"{sensor}_rate_per_min"] = (diff / dt * 60.0).fillna(0.0)

    if USE_SYSTEM_STATE_FEATURES and "system_state" in df.columns:
        state_dummies = pd.get_dummies(df["system_state"], prefix="state")
        feat = pd.concat([feat, state_dummies], axis=1)

    feat["elapsed_min"] = df.groupby("trajectory_id")["timestamp_sec"].transform(
        lambda x: (x - x.min()) / 60.0)

    if satellite_features and SATELLITE_AS_MODEL_FEATURES:
        for k, v in satellite_features.items():
            feat[k] = v

    feat["trajectory_id"] = df["trajectory_id"].values
    if "split" in df.columns:
        feat["split"] = df["split"].values
    for t in CLF_TARGETS + [REG_TARGET]:
        if t in df.columns:
            feat[t] = df[t].values
    return feat


# ---------------------------------------------------------------------------
# Step 4: train (identical hyperparameters/early stopping to your notebook)
# ---------------------------------------------------------------------------

def make_clf(scale_pos_weight):
    return XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0, gamma=0.5,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        tree_method="hist", random_state=SEED, n_jobs=-1,
        early_stopping_rounds=40,
    )


def make_reg():
    return XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
        eval_metric="mae", tree_method="hist", random_state=SEED, n_jobs=-1,
        early_stopping_rounds=40,
    )


def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = np.nanargmax(f1[:-1]) if len(thr) else 0
    return (thr[i] if len(thr) else 0.5), f1[i] if len(thr) else 0.0


def main():
    telemetry_csv = find_input_file("synthetic_glacier_telemetry.csv")
    yearly_csv = find_input_file("yearly_summary.csv")
    scenes_csv = find_input_file("scenes.csv")
    assert telemetry_csv, "synthetic_glacier_telemetry.csv not found"

    raw = load_and_prepare_raw(telemetry_csv)
    print(f"Loaded {raw.shape[0]} rows, {raw['trajectory_id'].nunique()} trajectories")

    satellite_features = None
    if scenes_csv:
        satellite_features = build_satellite_features(yearly_csv, scenes_csv)
        print("Satellite site summary:", satellite_features)
        print(f"Satellite as model features: {SATELLITE_AS_MODEL_FEATURES} "
              "(satellite is applied after the model by satellite_risk.py)")
    else:
        print("NOTE: scenes.csv not found -- satellite summary skipped.")

    pm = build_features(raw, satellite_features)
    FEATURES = [c for c in pm.columns
                if c not in (["trajectory_id", "split"] + CLF_TARGETS + [REG_TARGET])]
    pm[FEATURES] = pm[FEATURES].fillna(0.0)
    dead = [c for c in FEATURES if pm.loc[pm["split"] == "TRAIN", c].nunique() <= 1]
    print(f"Constant (zero-information) features in TRAIN: {len(dead)} -> "
          f"{[d for d in dead if 'seismic' in d or 'stale' in d][:8]}{' ...' if len(dead) > 8 else ''}")
    print(f"Feature matrix: {pm.shape}, {len(FEATURES)} features "
          f"(was 6-sensor before, now {len(RAW_SENSORS)}-sensor incl. seismic)")

    splits = {s: pm[pm["split"] == s].reset_index(drop=True)
              for s in ["TRAIN", "VAL", "TEST_UNSEEN"]}
    X_train, X_val, X_test = (splits["TRAIN"][FEATURES], splits["VAL"][FEATURES],
                               splits["TEST_UNSEEN"][FEATURES])

    models, baselines = {}, {}
    for tgt, tag in zip(CLF_TARGETS, ["gradual", "sudden"]):
        y_train = splits["TRAIN"][tgt].values
        y_val = splits["VAL"][tgt].values
        spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        clf = make_clf(spw)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models[tag] = clf
        lr = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced",
                                                C=1.0, random_state=SEED))
        lr.fit(X_train, y_train)
        baselines[tag] = lr
        print(f"[{tag}] scale_pos_weight={spw:.1f}  best_iteration={clf.best_iteration}  "
              f"best_val_aucpr={clf.best_score:.4f}")

    rt_train = splits["TRAIN"][REG_TARGET].notna().values
    rt_val = splits["VAL"][REG_TARGET].notna().values
    reg = make_reg()
    reg.fit(X_train[rt_train], splits["TRAIN"].loc[rt_train, REG_TARGET],
            eval_set=[(X_val[rt_val], splits["VAL"].loc[rt_val, REG_TARGET])], verbose=False)
    models["ttc"] = reg
    print(f"[time_to_critical_min] best_iteration={reg.best_iteration}  "
          f"best_val_mae={reg.best_score:.2f} min")

    print("\n--- Evaluation ---")
    metrics_out = {}
    for tgt, tag in zip(CLF_TARGETS, ["gradual", "sudden"]):
        metrics_out[tag] = {}
        clf = models[tag]
        y_val = splits["VAL"][tgt].values
        p_val = clf.predict_proba(X_val)[:, 1]
        thr, _ = best_f1_threshold(y_val, p_val)
        for split_name in ["VAL", "TEST_UNSEEN"]:
            part = splits[split_name]
            y_true = part[tgt].values
            p = clf.predict_proba(part[FEATURES])[:, 1]
            y_pred = (p >= thr).astype(int)
            pr_auc = average_precision_score(y_true, p) if y_true.sum() > 0 else float("nan")
            roc_auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            print(f"  [{tgt}][{split_name}] PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  "
                  f"acc@thr={acc:.4f}  F1@thr={f1:.4f}  thr={thr:.3f}")
            print(f"    confusion [[TN,FP],[FN,TP]]:\n{cm}")
            metrics_out[tag][split_name] = {
                "pr_auc": None if np.isnan(pr_auc) else float(pr_auc),
                "roc_auc": None if np.isnan(roc_auc) else float(roc_auc),
                "accuracy": float(acc),
                "f1": float(f1),
                "threshold": float(thr),
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
                "positive_count": int(y_true.sum()),
            }

    metrics_out["time_to_critical"] = {
        "val_mae_min": float(mean_absolute_error(
            splits["VAL"].loc[rt_val, REG_TARGET],
            reg.predict(X_val[rt_val])
        ))
    }
    metrics_out["notes"] = [
        "Metrics are evaluated on trajectory-grouped synthetic telemetry.",
        "TEST_UNSEEN sudden-impact has no positive examples, so PR-AUC/ROC-AUC/F1 are not meaningful there.",
        "Satellite is applied as a post-model rule-based prior, not as a trained XGBoost feature.",
        "Seismic energy is currently zero-filled in historical training data and therefore cannot be learned meaningfully yet."
    ]
    with open(f"{OUT_DIR}/metrics.json", "w") as fh:
        json.dump(metrics_out, fh, indent=2)

    for tag in ["gradual", "sudden"]:
        models[tag].save_model(f"{OUT_DIR}/glacierguard_{tag}.json")
    models["ttc"].save_model(f"{OUT_DIR}/glacierguard_ttc.json")

    with open(f"{OUT_DIR}/features.json", "w") as fh:
        json.dump(FEATURES, fh, indent=2)
    with open(f"{OUT_DIR}/raw_sensors.json", "w") as fh:
        json.dump({"RAW_SENSORS": RAW_SENSORS, "VALID_PAIRING": VALID_PAIRING,
                    "SHORT_WIN": SHORT_WIN, "LONG_WIN": LONG_WIN}, fh, indent=2)
    if satellite_features:
        with open(f"{OUT_DIR}/satellite_site_features.json", "w") as fh:
            json.dump(satellite_features, fh, indent=2)

    print(f"\nSaved models + schema to {OUT_DIR}/")
    print("\nSATELLITE NOTE: satellite data is a post-model prior (satellite_risk.py), "
          "not a trained feature, because synthetic trajectories have no real dates.")
    print("\nLIMITATION TO STATE IN YOUR DEMO: seismic_energy is backfilled as 0.0 "
          "in all historical training rows (no real seismic data existed when the "
          "synthetic trajectories were generated), so the model cannot yet weight "
          "seismic activity meaningfully. It's wired into the pipeline and will "
          "start contributing once training trajectories include real seismic variation.")


if __name__ == "__main__":
    main()