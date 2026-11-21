# Solar generation model: evaluation and deviation review

Selected model: **Random Forest** (chosen by validation MAE after comparing Extra Trees leaf sizes 1, 2, and 4, Random Forest, and Histogram Gradient Boosting). Target: plant-level sum of inverter AC power at each timestamp, in source dataset units.

The data contain 3,262 unique timestamps. The chronological holdout spans 2020-06-12 21:30:00 through 2020-06-17 23:45:00 (978 plant-timestamp rows); no timestamp in this period was used for fitting or model selection.

## Held-out performance

| Scope | MAE | RMSE | R² |
|---|---:|---:|---:|
| Both plants | 466.8 | 1,309.3 | 0.9657 |
| Plant 1 | 228.3 | 503.0 | 0.9962 |
| Plant 2 | 704.4 | 1,780.3 | 0.9001 |

Plant 2 has substantially higher held-out error than Plant 1, so the combined score should not be used as a substitute for plant-specific review.

## Deviation review

Using each plant's 99th-percentile absolute residual on the validation block as the review threshold produced **58 flagged test intervals**, grouped into **24 periods**. The thresholds were 806.0 for Plant 1 and 5,214.9 for Plant 2.

The flags are concentrated in daytime/high-irradiance conditions:
- **High irradiance (>0.5):** Plant 1: 39/94 rows flagged; Plant 2: 10/64 rows flagged.
- **09:00–15:00:** Plant 1: 44/120 rows flagged; Plant 2: 11/120 rows flagged.
- A simple sensor-range screen flagged 0 deviation rows for ambient/module temperatures or negative irradiation outside set bounds.
- Zero AC readings per timestamp averaged 1.3 on flagged rows versus 11.7 on unflagged rows. The unflagged set is dominated by near-zero-irradiance night intervals; this descriptive difference does not support a fault conclusion.
- These are associations in one short held-out period, not causal findings. Review source telemetry and operating context before drawing operational conclusions.

## Largest flagged periods

| Plant | Start | End | Intervals | Peak absolute residual | Mean irradiation | Mean zero AC readings |
|---|---|---|---:|---:|---:|---:|
| Plant 2 | 2020-06-16 14:15:00 | 2020-06-16 14:30:00 | 2 | 12,447.6 | 0.528 | 11.0 |
| Plant 2 | 2020-06-17 13:45:00 | 2020-06-17 13:45:00 | 1 | 11,911.3 | 0.579 | 0.0 |
| Plant 2 | 2020-06-13 10:30:00 | 2020-06-13 10:45:00 | 2 | 11,658.2 | 0.649 | 15.0 |
| Plant 2 | 2020-06-16 11:30:00 | 2020-06-16 11:30:00 | 1 | 9,625.5 | 0.768 | 0.0 |
| Plant 2 | 2020-06-13 09:45:00 | 2020-06-13 09:45:00 | 1 | 9,235.4 | 0.607 | 0.0 |
| Plant 2 | 2020-06-14 12:45:00 | 2020-06-14 12:45:00 | 1 | 6,050.3 | 0.885 | 8.0 |
| Plant 2 | 2020-06-16 12:45:00 | 2020-06-16 13:15:00 | 3 | 5,890.1 | 0.563 | 0.0 |
| Plant 1 | 2020-06-14 13:30:00 | 2020-06-14 14:15:00 | 3 | 3,240.2 | 0.848 | 1.3 |
| Plant 1 | 2020-06-14 09:15:00 | 2020-06-14 12:45:00 | 10 | 2,461.8 | 0.601 | 1.0 |
| Plant 1 | 2020-06-13 10:45:00 | 2020-06-13 12:15:00 | 6 | 2,447.5 | 0.870 | 0.0 |

## Scope and limitations

Predictors are weather sensor readings (ambient temperature, module temperature, irradiation), cyclic clock/calendar features, and plant identity. AC/DC generation readings, inverter zero counts, and yield values were not used as model predictors. The dataset covers only about a month, so the held-out results represent a short future window rather than seasonal generalization. A large residual is a model discrepancy to investigate, not a diagnosis of equipment failure.
