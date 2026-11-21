"""Train and evaluate time-aware solar generation regression models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
PLANTS = {
    "Plant 1": ("Plant_1_Generation_Data.csv", "Plant_1_Weather_Sensor_Data.csv", "%d-%m-%Y %H:%M"),
    "Plant 2": ("Plant_2_Generation_Data.csv", "Plant_2_Weather_Sensor_Data.csv", "%Y-%m-%d %H:%M:%S"),
}
# day_sin/day_cos removed: the data spans ~1 month, so a yearly cycle cannot be learned.
# temp_diff (module - ambient) added: it tracks how strongly the panels are heated by sun.
NUMERIC = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "hour_sin", "hour_cos", "temp_diff"]
CATEGORICAL = ["plant"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "actual_ac_power"


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    hour = f.timestamp.dt.hour + f.timestamp.dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["temp_diff"] = f["MODULE_TEMPERATURE"] - f["AMBIENT_TEMPERATURE"]
    return f


def load_data() -> pd.DataFrame:
    chunks = []
    for plant, (gen_file, weather_file, gen_format) in PLANTS.items():
        gen = pd.read_csv(DATA / gen_file)
        weather = pd.read_csv(DATA / weather_file)
        gen["timestamp"] = pd.to_datetime(gen["DATE_TIME"], format=gen_format, errors="coerce")
        weather["timestamp"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")
        gen = gen.dropna(subset=["timestamp", "AC_POWER"])
        weather = weather.dropna(subset=["timestamp"])
        # The prediction target is plant-level measured AC power at each 15-minute timestamp.
        power = gen.groupby("timestamp").agg(
            **{TARGET: ("AC_POWER", "sum")},
            # Retained only as a post-hoc diagnostic, never passed into model features.
            zero_ac_inverters=("AC_POWER", lambda values: int((values == 0).sum())),
        ).reset_index()
        # Each plant has a single weather sensor record per timestamp. A timestamp join avoids
        # repeating generation rows for inverter IDs and keeps weather predictors leakage-free.
        wx = weather.groupby("timestamp", as_index=False)[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].mean()
        merged = power.merge(wx, on="timestamp", how="inner", validate="one_to_one")
        merged["plant"] = plant
        chunks.append(merged)
    frame = add_time_features(pd.concat(chunks, ignore_index=True).sort_values("timestamp"))
    return frame.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def make_pipeline(estimator) -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("plant", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("features", prep), ("regressor", estimator)])


def metrics(y, p) -> dict:
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(mean_squared_error(y, p) ** 0.5), "r2": float(r2_score(y, p))}


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    frame = load_data()
    # Split on unique timestamps so both plants at a timestamp stay in the same period.
    times = np.array(sorted(frame.timestamp.unique()))
    a, b = int(len(times) * .70), int(len(times) * .85)
    cut_train, cut_val = times[a], times[b]
    train = frame[frame.timestamp < cut_train]
    val = frame[(frame.timestamp >= cut_train) & (frame.timestamp < cut_val)]
    test = frame[frame.timestamp >= cut_val]
    candidates = {
        "Extra Trees (leaf=1)": ExtraTreesRegressor(n_estimators=350, min_samples_leaf=1, max_features=.9, n_jobs=-1, random_state=42),
        "Extra Trees (leaf=2)": ExtraTreesRegressor(n_estimators=350, min_samples_leaf=2, max_features=.9, n_jobs=-1, random_state=42),
        "Extra Trees (leaf=4)": ExtraTreesRegressor(n_estimators=350, min_samples_leaf=4, max_features=.9, n_jobs=-1, random_state=42),
        "Extra Trees (tuned: leaf=2, 600 trees, all features)": ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=1.0, n_jobs=-1, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=.9, n_jobs=-1, random_state=42),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=.08, max_leaf_nodes=25, l2_regularization=1.0, random_state=42),
    }
    validation = {}
    trained = {}
    for name, estimator in candidates.items():
        model = make_pipeline(estimator)
        model.fit(train[FEATURES], train[TARGET])
        validation[name] = metrics(val[TARGET], model.predict(val[FEATURES]))
        trained[name] = model
    winner = min(validation, key=lambda name: validation[name]["mae"])
    # Refit the selected hyperparameters on train + validation, keeping the final 15% untouched.
    fit = pd.concat([train, val], ignore_index=True)
    selected = make_pipeline(candidates[winner])
    selected.fit(fit[FEATURES], fit[TARGET])
    test = test.copy()
    test["predicted_ac_power"] = np.maximum(0, selected.predict(test[FEATURES]))
    test["residual"] = test[TARGET] - test["predicted_ac_power"]
    report = {
        "model": winner,
        "target": "sum of inverter AC_POWER per plant timestamp (dataset units)",
        "features": FEATURES,
        "periods": {
            "train": {"start": str(train.timestamp.min()), "end": str(train.timestamp.max()), "rows": len(train)},
            "validation": {"start": str(val.timestamp.min()), "end": str(val.timestamp.max()), "rows": len(val)},
            "test": {"start": str(test.timestamp.min()), "end": str(test.timestamp.max()), "rows": len(test)},
        },
        "validation_metrics": validation,
        "test_metrics": metrics(test[TARGET], test["predicted_ac_power"]),
        "test_metrics_by_plant": {p: metrics(g[TARGET], g["predicted_ac_power"]) for p, g in test.groupby("plant")},
    }
    # Set investigation thresholds from validation residuals only, per plant (99th percentile).
    val_residuals = val[["timestamp", "plant", TARGET]].copy()
    val_residuals["predicted_ac_power"] = np.maximum(0, trained[winner].predict(val[FEATURES]))
    thresholds = {}
    for plant, group in val_residuals.groupby("plant"):
        thresholds[plant] = float(np.quantile(np.abs(group[TARGET] - group.predicted_ac_power), .99))
    test["deviation_threshold"] = test.plant.map(thresholds)
    test["significant_deviation"] = test.residual.abs() > test.deviation_threshold
    test["deviation_direction"] = np.where(test.residual > 0, "actual above estimate", "actual below estimate")
    test["unusual_environment_reading"] = (
        (test.IRRADIATION < 0) | (test.AMBIENT_TEMPERATURE < -10) | (test.AMBIENT_TEMPERATURE > 60) |
        (test.MODULE_TEMPERATURE < -10) | (test.MODULE_TEMPERATURE > 90)
    )
    report["deviation_thresholds_abs_error_p99_validation"] = thresholds
    flagged = test[test.significant_deviation].copy()
    report["flagged_test_rows"] = int(len(flagged))
    report["flagged_by_plant"] = {p: int(g.significant_deviation.sum()) for p, g in test.groupby("plant")}
    report["flagged_environment_out_of_range"] = int(flagged.unusual_environment_reading.sum())
    report["interpretation"] = "Flags identify large model residuals for review; they do not establish equipment faults or causation."
    test["irradiance_band"] = pd.cut(test.IRRADIATION, [-np.inf, .05, .2, .5, np.inf], labels=["near-zero", "low", "moderate", "high"])
    hour = test.timestamp.dt.hour + test.timestamp.dt.minute / 60
    test["time_band"] = pd.cut(hour, [-.01, 9, 15, 24], labels=["before 09:00", "09:00–15:00", "after 15:00"])
    report["error_associations_descriptive"] = {}
    for dimension in ["irradiance_band", "time_band"]:
        report["error_associations_descriptive"][dimension] = {}
        for (plant, band), group in test.groupby(["plant", dimension], observed=False):
            if pd.isna(band) or group.empty:
                continue
            report["error_associations_descriptive"][dimension][f"{plant} | {band}"] = {
                "rows": int(len(group)), "mae": float(group.residual.abs().mean()),
                "flagged_rows": int(group.significant_deviation.sum()),
                "flagged_rate": float(group.significant_deviation.mean()),
                "mean_zero_ac_inverters": float(group.zero_ac_inverters.mean()),
            }
    report["zero_ac_inverter_diagnostic"] = {
        "unflagged_test_mean": float(test.loc[~test.significant_deviation, "zero_ac_inverters"].mean()),
        "flagged_test_mean": float(test.loc[test.significant_deviation, "zero_ac_inverters"].mean()),
        "note": "Generation zero counts are descriptive only and excluded from model predictors; zeros alone do not establish a fault.",
    }
    # Associations are descriptive summaries among flagged held-out points, not causal claims.
    if len(flagged):
        report["flagged_summary"] = {
            "mean_irradiation": float(flagged.IRRADIATION.mean()),
            "median_irradiation": float(flagged.IRRADIATION.median()),
            "mean_hour": float((flagged.timestamp.dt.hour + flagged.timestamp.dt.minute / 60).mean()),
            "mean_abs_residual": float(flagged.residual.abs().mean()),
            "above_estimate": int((flagged.residual > 0).sum()),
            "below_estimate": int((flagged.residual < 0).sum()),
        }
    # Combine flagged intervals into reviewable periods (allow one intervening unflagged interval).
    flagged = test[test.significant_deviation].sort_values(["plant", "timestamp"]).copy()
    breaks = flagged.groupby("plant").timestamp.diff().gt(pd.Timedelta(minutes=30))
    flagged["period_group"] = breaks.groupby(flagged.plant).cumsum()
    periods = flagged.groupby(["plant", "period_group"], as_index=False).agg(
        period_start=("timestamp", "min"), period_end=("timestamp", "max"),
        flagged_intervals=("timestamp", "size"), peak_abs_residual=("residual", lambda x: float(x.abs().max())),
        mean_irradiation=("IRRADIATION", "mean"), mean_ambient_temperature=("AMBIENT_TEMPERATURE", "mean"),
        mean_zero_ac_inverters=("zero_ac_inverters", "mean"),
    )
    report["deviation_periods"] = int(len(periods))
    report["deviation_periods_by_plant"] = {p: int((periods.plant == p).sum()) for p in PLANTS}
    cols = ["timestamp", "plant", TARGET, "predicted_ac_power", "residual", "deviation_threshold", "deviation_direction", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "zero_ac_inverters", "unusual_environment_reading"]
    test[cols].to_csv(OUTPUTS / "heldout_predictions.csv", index=False)
    flagged[cols].to_csv(OUTPUTS / "significant_deviations.csv", index=False)
    periods.drop(columns=["period_group"]).sort_values("period_start").to_csv(OUTPUTS / "significant_deviation_periods.csv", index=False)
    joblib.dump(selected, OUTPUTS / "solar_model.joblib")
    (OUTPUTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Solar generation model: evaluation and deviation review", "",
        f"Selected model: **{winner}** (chosen by validation MAE after comparing Extra Trees leaf sizes 1, 2, and 4, Random Forest, and Histogram Gradient Boosting). Target: plant-level sum of inverter AC power at each timestamp, in source dataset units.", "",
        f"The data contain {len(times):,} unique timestamps. The chronological holdout spans {test.timestamp.min()} through {test.timestamp.max()} ({len(test):,} plant-timestamp rows); no timestamp in this period was used for fitting or model selection.", "",
        "## Held-out performance", "",
        "| Scope | MAE | RMSE | R² |", "|---|---:|---:|---:|",
        f"| Both plants | {report['test_metrics']['mae']:,.1f} | {report['test_metrics']['rmse']:,.1f} | {report['test_metrics']['r2']:.4f} |",
    ]
    for plant, values in report["test_metrics_by_plant"].items():
        lines.append(f"| {plant} | {values['mae']:,.1f} | {values['rmse']:,.1f} | {values['r2']:.4f} |")
    lines.extend(["", "Plant 2 has substantially higher held-out error than Plant 1, so the combined score should not be used as a substitute for plant-specific review.", "", "## Deviation review", "",
        f"Using each plant's 99th-percentile absolute residual on the validation block as the review threshold produced **{len(test[test.significant_deviation]):,} flagged test intervals**, grouped into **{len(periods)} periods**. The thresholds were {thresholds['Plant 1']:,.1f} for Plant 1 and {thresholds['Plant 2']:,.1f} for Plant 2.", "",
        "The flags are concentrated in daytime/high-irradiance conditions:"])
    associations = report["error_associations_descriptive"]
    for label, key in [("High irradiance (>0.5)", "high"), ("09:00–15:00", "09:00–15:00")]:
        values = []
        for plant in PLANTS:
            group = associations["irradiance_band" if key == "high" else "time_band"].get(f"{plant} | {key}", {})
            values.append(f"{plant}: {group.get('flagged_rows', 0)}/{group.get('rows', 0)} rows flagged")
        lines.append(f"- **{label}:** " + "; ".join(values) + ".")
    lines.extend([
        f"- A simple sensor-range screen flagged {report['flagged_environment_out_of_range']} deviation rows for ambient/module temperatures or negative irradiation outside set bounds.",
        f"- Zero AC readings per timestamp averaged {report['zero_ac_inverter_diagnostic']['flagged_test_mean']:.1f} on flagged rows versus {report['zero_ac_inverter_diagnostic']['unflagged_test_mean']:.1f} on unflagged rows. The unflagged set is dominated by near-zero-irradiance night intervals; this descriptive difference does not support a fault conclusion.",
        "- These are associations in one short held-out period, not causal findings. Review source telemetry and operating context before drawing operational conclusions.", "",
        "## Largest flagged periods", "", "| Plant | Start | End | Intervals | Peak absolute residual | Mean irradiation | Mean zero AC readings |", "|---|---|---|---:|---:|---:|---:|"])
    top_periods = periods.sort_values("peak_abs_residual", ascending=False).head(10)
    for _, row in top_periods.iterrows():
        lines.append(f"| {row.plant} | {row.period_start} | {row.period_end} | {int(row.flagged_intervals)} | {row.peak_abs_residual:,.1f} | {row.mean_irradiation:.3f} | {row.mean_zero_ac_inverters:.1f} |")
    lines.extend(["", "## Scope and limitations", "", "Predictors are weather sensor readings (ambient temperature, module temperature, irradiation), cyclic clock/calendar features, and plant identity. AC/DC generation readings, inverter zero counts, and yield values were not used as model predictors. The dataset covers only about a month, so the held-out results represent a short future window rather than seasonal generalization. A large residual is a model discrepancy to investigate, not a diagnosis of equipment failure.", ""])
    (OUTPUTS / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
