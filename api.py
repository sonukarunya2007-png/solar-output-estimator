"""FastAPI service and lightweight dashboard for the trained solar model."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from train_model import add_time_features, FEATURES, ROOT

app = FastAPI(title="Solar Output Estimator", description="Plant-level AC power estimates with held-out deviation review.", version="1.0.0")
MODEL_PATH = ROOT / "solar_model.joblib"
REPORT_PATH = ROOT / "metrics.json"
PREDICTIONS_PATH = ROOT / "heldout_predictions.csv"
DEVIATIONS_PATH = ROOT / "significant_deviation_periods.csv"
STATIC = ROOT

class PredictionInput(BaseModel):
    plant: str = Field(pattern=r"^Plant [12]$")
    timestamp: datetime
    ambient_temperature: float = Field(ge=-40, le=70)
    module_temperature: float = Field(ge=-40, le=120)
    irradiation: float = Field(ge=0)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ready", "model_available": MODEL_PATH.exists() and REPORT_PATH.exists()}


def load_report():
    if not REPORT_PATH.exists():
        raise HTTPException(503, "Model artifacts are missing. Run python train_model.py first.")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/api/summary")
def summary():
    return load_report()


@app.get("/api/predictions")
def predictions(plant: str | None = Query(default=None, pattern=r"^Plant [12]$"), limit: int = Query(default=500, ge=1, le=3000)):
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(503, "Held-out prediction file is missing. Run python train_model.py first.")
    data = pd.read_csv(PREDICTIONS_PATH, parse_dates=["timestamp"])
    if plant:
        data = data[data.plant == plant]
    data = data.sort_values("timestamp").tail(limit)
    data["timestamp"] = data.timestamp.dt.strftime("%Y-%m-%d %H:%M")
    return data.replace({np.nan: None}).to_dict(orient="records")


@app.get("/api/deviations")
def deviations(plant: str | None = Query(default=None, pattern=r"^Plant [12]$"), limit: int = Query(default=100, ge=1, le=1000)):
    if not DEVIATIONS_PATH.exists():
        return []
    data = pd.read_csv(DEVIATIONS_PATH)
    if plant:
        data = data[data.plant == plant]
    data = data.sort_values("period_start", ascending=False).head(limit)
    data["period_start"] = data.period_start.str.slice(0, 16)
    data["period_end"] = data.period_end.str.slice(0, 16)
    return data.replace({np.nan: None}).to_dict(orient="records")


@app.post("/api/predict")
def predict(payload: PredictionInput):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model artifact is missing. Run python train_model.py first.")
    t = payload.timestamp.replace(tzinfo=None)  # treat as plant-local clock time; never shift to UTC
    row = pd.DataFrame([{
        "timestamp": t,
        "plant": payload.plant,
        "AMBIENT_TEMPERATURE": payload.ambient_temperature,
        "MODULE_TEMPERATURE": payload.module_temperature,
        "IRRADIATION": payload.irradiation,
    }])
    row = add_time_features(row)
    estimate = max(0.0, float(joblib.load(MODEL_PATH).predict(row[FEATURES])[0]))
    return {"plant": payload.plant, "timestamp": t.isoformat(), "estimated_ac_power": estimate, "units": "dataset AC_POWER units", "note": "Estimate from environmental and time features; not a fault diagnosis."}

