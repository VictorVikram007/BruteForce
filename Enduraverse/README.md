# Battery Intelligence System for EV Fleets

A professional-grade ML system designed to reduce warranty risk and enable predictive maintenance for electric vehicle fleets.

## 🚀 Business Value
- **Warranty Risk Mitigation**: Predict battery failures before they occur, allowing for optimized warranty reserves.
- **Predictive Maintenance**: Enables fleet operators to schedule replacements during planned downtime, avoiding operational disruptions.
- **Fleet-Level Risk Tracking**: Aggregate analytics to monitor global fleet health and identify high-risk assets.

## 🔬 Physics-Informed Design
This system uses a **multi-phase degradation model** anchored to **Lithium Iron Phosphate (LFP)** battery characteristics (industry-standard ~3,500 cycle life, 80% EOL threshold).

### Modeling Key Factors:
- **Temperature Acceleration**: Arrhenius-inspired exponential aging models.
- **DoD Stress**: Non-linear impact of deep cycling on active material loss.
- **C-Rate Impact**: High rates are mapped to lithium plating and thermal stress risks.
- **Elite 4-Phase Degradation**:
  1. *Formation Phase*: Ultra-slow early fade (>95% SOH).
  2. *Stabilization Phase*: SEI layer stabilization (slow aging).
  3. *Linear Phase*: Normal operational degradation.
  4. *Avalanche Phase*: Accelerated material loss (1.6x rapid drop near 80% EOL).

## 🧠 Technical Stack
- **Modular Pipeline**: Scikit-Learn pipelines for consistent scaling and inference.
- **Uncertainty Layer**: Estimator variance tracking to flag unreliable predictions (± range).
- **Temporal Memory**: 20-cycle rolling windows for usage history.
- **Benchmarking**: Proven superiority of RandomForest over Linear baselines in capturing non-linear physics.

## 🛠️ Usage

### 1. Training & Analysis
Run the full system orchestrator to generate data, train the model, and run diagnostic analytics:
```bash
python main.py
```

### 2. Launching the Dashboard
Interact with the battery intelligence dashboard for "What-if" simulations:
```bash
streamlit run app.py
```

## ⚠️ Failure Case Analysis
The model is designed to be self-aware. It flags high uncertainty in:
- **Edge Regions**: Extreme temperature (>50°C) or high DoD (>95%) where training data is sparse.
- **Domain Shifts**: Predictions for operational regimes not covered in the calibration set.
- *Solution*: The uncertainty estimation layer (± cycles) provides a built-in reliability metric for critical decision-making.

---
*Developed for Top-Tier EV Fleet Intelligence Finals.*
