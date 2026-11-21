# Solar Output Estimator

Estimates plant-level AC power from weather sensor readings and timestamp for the two supplied solar plants. The app includes a time-ordered holdout evaluation and a review list for unusually large prediction residuals.

## Run locally

From this directory in PowerShell, create the environment and start the app using the included saved model:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The API documentation is at <http://127.0.0.1:8000/docs>.

To retrain the model and regenerate reports from the included source CSVs, run `\.venv\Scripts\python.exe train_model.py` before starting the app. The CSVs are in `data/`.

## Method

- Aggregate inverter AC_POWER by plant and timestamp; use environmental readings from the matching timestamp.
- Use ambient temperature, module temperature, irradiation, cyclic time-of-day/annual features, and plant identity as predictors. Generation values are used only to build the target and residuals.
- Tune Extra Trees leaf size (1, 2, 4) and compare against Random Forest and Histogram Gradient Boosting on a chronological validation block. The last 15% of unique timestamps is reserved as unseen test data.
- Choose the model by validation MAE, refit through the validation period, then report held-out metrics.
- Flag test residuals beyond the plant-specific 99th percentile of validation-period absolute residuals. This supports review; a flag alone does not identify cause or establish an equipment fault.

Adjacent flagged intervals are grouped into review periods, allowing one intervening unflagged interval. See `outputs/analysis.md` for the evaluation and investigation summary, `outputs/metrics.json` for machine-readable metrics, `outputs/solar_model.joblib` for the saved model, and the held-out prediction/deviation CSVs in `outputs/` after training.
