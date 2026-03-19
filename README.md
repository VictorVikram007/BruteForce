# Battery Intelligence System for EV Fleets

### Predict Battery Failure Before It Happens using Physics-Informed AI

---

## Overview

The Battery Intelligence System is a hybrid AI platform that predicts the Remaining Useful Life (RUL) of lithium-ion batteries in electric vehicle fleets.

It combines physics-based degradation modeling with machine learning to deliver:

* Accurate predictions
* Confidence-aware decisions
* Real-world fleet insights

---

## Problem Statement

* EV battery failures are unpredictable
* High replacement costs
* Fleet downtime leads to operational losses
* No reliable system for early failure detection

---

## Solution

A Physics-Informed Hybrid AI System that:

* Predicts RUL (cycles, months, energy)
* Combines physics-based and machine learning models
* Provides confidence score for reliability
* Incorporates real-world driving conditions (MIDC)
* Generates actionable insights for fleet operators

---

## How It Works

### Physics-Based Degradation

```python
cycle_stress = DoD**1.3 * C_rate**1.1
calendar_stress = exp((T - 25)/10) * (calendar_age_days / 365)

total_degradation = cycle_stress + 0.3 * calendar_stress
```

---

### SOH Degradation Model

```python
SOH(t) = SOH0 * exp(-k * t)
```

```python
k = 0.00008 + 0.12 * degradation_rate + 0.00025 * total_degradation
```

---

### Hybrid RUL Prediction

```python
RUL_hybrid = (1/k) * log(SOH_current / 0.8)
RUL_mean = (RUL_direct + RUL_hybrid) / 2
```

---

### Confidence Estimation

```python
confidence = 1 - (RUL_upper - RUL_lower) / max(RUL_mean, 1)
```

---

## Model Performance (9 Models)

| # | Model                | Type     | Accuracy (%) | RMSE (cycles) |
| - | -------------------- | -------- | ------------ | ------------- |
| 1 | Linear Regression    | Standard | 96.08        | 119.03        |
| 2 | Random Forest        | Standard | 99.46        | 44.23         |
| 3 | Gradient Boosting    | Standard | 99.17        | 54.86         |
| 4 | XGBoost              | Standard | 99.56        | 39.94         |
| 5 | LightGBM             | Standard | 99.86        | 22.49         |
| 6 | CatBoost             | Standard | 99.50        | 42.69         |
| 7 | SVR                  | Standard | 97.01        | 103.90        |
| 8 | PhyFuse-RUL (Hybrid) | Hybrid   | 99.89        | 11.19         |
| 9 | XGB-QRE Engine       | Hybrid   | 99.44        | 44.16         |

Best Model: PhyFuse-RUL

* ~50% lower error than best standard model
* Combines physics and ML for improved accuracy

---

## Feature Importance Insights

### Top Drivers (Across Models)

* Battery_ID
* Present_Capacity
* Cycle_Index
* State_of_Health

Battery lifecycle indicators dominate predictions.

---

### Hybrid Model Key Drivers (PhyFuse-RUL)

1. capacity_rolling_mean_10
2. capacity_rolling_mean_50
3. degradation_phase_code
4. calendar_age_days
5. capacity_fade

Rolling capacity trends are the strongest predictors.

---

### Key Takeaways

* Lifecycle and capacity trends are the strongest signals
* Temperature and DoD significantly impact degradation
* Hybrid modeling improves long-term prediction accuracy

---

## Dashboard Capabilities

* RUL prediction with confidence range
* Degradation trend visualization
* Sensitivity analysis (temperature, DoD, C-rate)
* Model comparison leaderboard
* Hybrid vs baseline performance comparison
* Fleet health monitoring

---

## Business Impact

* Reduced battery replacement costs
* Prevention of unexpected failures
* Improved fleet utilization
* Predictive maintenance enablement
* Scalable analytics platform for EV operations

---

## Example Output

* Remaining Life: 1100 cycles (~18 months)
* Confidence: High (±120 cycles)
* Status: Low Risk

---

## Project Structure

```bash
├── data_preprocessing.py
├── feature_engineering.py
├── model_training.py
├── evaluation.py
├── app.py
├── artifacts/
│   ├── battery_rul_hybrid_artifacts.joblib
│   └── test_predictions.csv
```

---

## Installation

```bash
git clone https://github.com/your-username/battery-rul-system.git
cd battery-rul-system

pip install -r requirements.txt
```

---

## Usage

### Train Models

```bash
python model_training.py
```

### Run Dashboard

```bash
streamlit run app.py
```

---

## Technologies Used

* Python
* Scikit-learn
* LightGBM / XGBoost / CatBoost
* Streamlit
* NumPy / Pandas / Matplotlib

---

## Future Improvements

* Real-time BMS integration
* Time-series modeling (LSTM)
* Cloud deployment for fleet analytics
* Digital twin simulation

---

## Why This Project Stands Out

This is a complete battery intelligence platform that integrates:

* Physics-based modeling and machine learning
* Explainability and uncertainty estimation
* Engineering insights with business impact

---

## Author

Team ButeForce

## License

This project is for educational and research purposes.
