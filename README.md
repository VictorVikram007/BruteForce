# RUL Model Performance Table (7 Standard + 2 Hybrid)

This table lists all 9 trained models with accuracy and error metrics.

- Accuracy is shown as `R2 x 100` (percentage).
- Error is shown as `RMSE` (cycles).
- Source: `rul-dashboard-backend/data/multi-model-results.json`

| # | Model Name | Family | Accuracy (%) | Error (RMSE cycles) |
|---|---|---|---:|---:|
| 1 | Linear Regression | Standard | 96.08 | 119.03 |
| 2 | Random Forest | Standard | 99.46 | 44.23 |
| 3 | Gradient Boosting | Standard | 99.17 | 54.86 |
| 4 | XGBoost | Standard | 99.56 | 39.94 |
| 5 | LightGBM | Standard | 99.86 | 22.49 |
| 6 | CatBoost | Standard | 99.50 | 42.69 |
| 7 | SVR | Standard | 97.01 | 103.90 |
| 8 | PhyFuse-RUL | Hybrid | 99.89 | 11.19 |
| 9 | XGB-QRE Engine | Hybrid | 99.44 | 44.16 |

## Notes

- Lower RMSE means lower prediction error.
- Higher accuracy percentage means better goodness-of-fit.
- Best overall error here is **PhyFuse-RUL**.

## Feature Evaluation

Feature importance was extracted from the latest training output in `rul-dashboard-backend/data/multi-model-results.json`.

### A) Aggregate Feature Importance (average across models with importance)

- Models included in this aggregate: 7
- Interpretation: larger average importance means stronger contribution to predictions in the standard feature space.

| Rank | Feature | Avg Importance | Models Contributing |
|---|---|---:|---:|
| 1 | Battery_ID | 500.167632 | 7 |
| 2 | Present_Capacity | 184.428021 | 6 |
| 3 | Cycle_Index | 107.731170 | 7 |
| 4 | State_of_Health | 19.284335 | 6 |
| 5 | Rolling_Avg_Temp | 9.427015 | 6 |
| 6 | Rolling_Avg_DoD | 3.869607 | 6 |
| 7 | Max_Discharge_C_Rate | 1.394180 | 6 |
| 8 | Daily_DoD | 1.343591 | 6 |
| 9 | Peak_Cell_Temp | 1.199767 | 6 |
| 10 | Avg_Ambient_Temp | 1.176327 | 6 |

### B) Best Model (PhyFuse-RUL) Top Feature Drivers

| Rank | Feature | Importance |
|---|---|---:|
| 1 | capacity_rolling_mean_10 | 0.500587 |
| 2 | capacity_rolling_mean_50 | 0.461615 |
| 3 | degradation_phase_code | 0.037277 |
| 4 | calendar_age_days | 0.000204 |
| 5 | capacity_fade | 0.000203 |
| 6 | cycle_index | 0.000092 |
| 7 | dod_rolling_mean_50 | 0.000010 |
| 8 | low_soh_flag | 0.000007 |
| 9 | calendar_stress | 0.000004 |
| 10 | avg_temp_rolling_mean_50 | 0.000001 |

### C) Quick Takeaways

- Battery lifecycle indicators dominate predictions (Battery_ID, Cycle_Index, Present_Capacity, State_of_Health).
- Thermal and usage stress features contribute meaningfully (Rolling_Avg_Temp, Daily_DoD, C-rate features).
- In the best hybrid model, rolling capacity trends are the strongest signal drivers.
