GLACIERGUARD FINAL RUN GUIDE
============================

1) Install dependencies
   python -m pip install -r requirements.txt

2) Train the final model
   python train_model_v3.py

   This automatically removes malformed rows with missing trajectory_id,
   timestamp_sec or split, checks trajectory-level split leakage, and writes:
     glacierguard_v3_outputs/glacierguard_gradual.json
     glacierguard_v3_outputs/glacierguard_sudden.json
     glacierguard_v3_outputs/glacierguard_ttc.json
     glacierguard_v3_outputs/features.json
     glacierguard_v3_outputs/metrics.json

3) Start the dashboard + streaming prototype
   python run_demo.py --tick 5

   Dashboard:
   http://localhost:8000/dashboard.html

   The demo uses live weather/seismic sources where available and simulated
   water-level/vibration/tilt/turbidity channels for prototype testing.
   The virtual telemetry clock advances by 10 minutes per generated row.

4) Optional unseen-test scenario replay
   python run_demo.py --scenario --tick 3

5) Satellite layer
   If satellite_data/scenes.csv is present, the Sentinel-2 layer is computed
   from the scenes dataset and used as a transparent post-model risk prior.
   It is NOT a trained XGBoost feature in this version.
   If scenes.csv is absent but a previous satellite_status.json exists, the
   dashboard keeps that cached context and labels it as cached.

SAFE PPT CLAIMS
===============
- Gradual escalation model, unseen synthetic trajectories:
  Accuracy 93.61%, PR-AUC 0.962, ROC-AUC 0.993, F1 0.779.
- Sudden-impact validation: PR-AUC 0.898, F1 0.900.
  Do NOT report TEST accuracy for sudden impact because TEST_UNSEEN has zero
  positive sudden-impact examples.
- Time-to-critical validation MAE: 72.9 minutes. Do not present this as a
  precise lead-time guarantee.
- Satellite: latest available Sentinel-2 observation / satellite context;
  do not call it real-time satellite data.
- Live feed: rainfall and temperature from Open-Meteo and seismic observations
  from USGS are integrated; water level, vibration, tilt and turbidity are
  currently simulated for prototype validation.
- Historical training seismic_energy is zero-filled, so the model has not yet
  learned a meaningful seismic contribution.

RESULTS FILE
============
glacierguard_v3_outputs/metrics.json contains the exact fresh evaluation.
