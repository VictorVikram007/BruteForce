import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from model_logic import split_data, predict_with_uncertainty

# Load data and model
print("Loading data...")
df = pd.read_csv("battery_dataset_final.csv")
print("Loading model...")
system_data = joblib.load("battery_intelligence_model.joblib")
pipeline = system_data['pipeline']

# Prepare data
print("Preparing test set...")
X_train, X_test, y_train, y_test, groups_test = split_data(df)

# Get predictions and uncertainty
print("Generating predictions...")
y_pred, std_pred = predict_with_uncertainty(pipeline, X_test)

# Plotting
print("Generating plot...")
plt.figure(figsize=(10, 6))
indices = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)

plt.errorbar(
    y_test.iloc[indices], y_pred[indices], 
    yerr=std_pred[indices], fmt='o', alpha=0.4, 
    ecolor='lightgray', elinewidth=1, capsize=0, label='Prediction ± Model Uncertainty'
)

ideal = [y_test.min(), y_test.max()]
plt.plot(ideal, ideal, 'r--', lw=2, label='Ideal Correlation')

plt.xlabel("Actual RUL (Cycles)")
plt.ylabel("Predicted RUL (Cycles)")
plt.title("RUL Prediction Accuracy & Diagnostic Uncertainty", fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Save image
save_path = "predicted_vs_actual_rul.png"
plt.savefig(save_path, dpi=120, bbox_inches='tight')
print(f"Plot saved to {os.path.abspath(save_path)}")
