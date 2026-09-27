# GlacierGuard Integrated Flow

This folder implements the complete prototype pipeline:

`7 sensors + satellite context -> preprocessing -> rolling features -> XGBoost models -> risk engine -> early warning`

## Main files

- `train_model_v3.py` — preprocessing, feature engineering, training and evaluation
- `live_inference.py` — applies the trained models to the live stream
- `risk_engine.py` — converts probabilities + satellite event + lead time into risk level and alert
- `realtime_feed.py` — live weather/seismic ingestion plus prototype sensor channels
- `satellite_risk.py` — Sentinel-2 context and post-model satellite prior
- `run_demo.py` — runs the complete pipeline and dashboard server
- `dashboard.html` — live dashboard
- `synthetic_glacier_telemetry.csv` — training/evaluation telemetry
- `glacierguard_v3_outputs/` — trained models, feature schema and metrics

## Commands

```text
pip install -r requirements.txt
python train_model_v3.py
python run_demo.py --scenario --tick 3
```

Then open:
`http://localhost:8000/dashboard.html`
