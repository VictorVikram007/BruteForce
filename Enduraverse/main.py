import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from data_generation import generate_battery_data, calculate_rul
from model_logic import split_data, train_quantile_models, predict_quantiles, evaluate_models
from visualization import plot_degradation_trends, plot_diagnostic_dashboard, plot_sensitivity_analysis

def run_overhaul_pipeline():
    """
    Orchestrates the upgraded ML pipeline:
    Kalman Filtering -> XGBoost Quantile Regression -> Advanced Diagnostics
    """
    print("🚀 Initializing Upgraded Battery Intelligence System...\n")
    
    # --- 1. Data Layer (Kalman Smoothing Included) ---
    print("[1/5] Generating LFP-anchored synthetic data (Smoothed)...")
    raw_df = generate_battery_data(num_batteries=60, cycles_per_battery=3600)
    df = calculate_rul(raw_df)
    df.to_csv("battery_dataset_final.csv", index=False)
    print(f"      - Dataset built: {len(df)} cycles across {df['Battery_ID'].nunique()} batteries.")
    
    # --- 2. Training XGBoost Ensemble ---
    print("\n[2/5] Training XGBoost Quantile Ensemble...")
    X_train, X_test, y_train, y_test, groups = split_data(df)
    models = train_quantile_models(X_train, y_train)
    
    # --- 3. Evaluation & Diagnostics ---
    print("\n[3/5] Evaluating Model Diagnostics...")
    preds = predict_quantiles(models, X_test)
    metrics = evaluate_models(y_test, preds)
    
    # --- 4. Persistence ---
    print("\n[4/5] Saving Intelligence System...")
    joblib.dump({
        'models': models,
        'features': X_train.columns.tolist(),
        'metrics': metrics
    }, "battery_intelligence_model.joblib")
    
    # --- 5. Visual Intelligence ---
    print("\n[5/5] Generating Advanced Diagnostic Visuals...")
    
    # Degradation Trends (Filtered)
    fig_trends = plot_degradation_trends(df)
    fig_trends.savefig("degradation_trends_filtered.png")
    
    # ML Performance Dashboard (The pivot fix)
    fig_diag = plot_diagnostic_dashboard(y_test, preds, metrics)
    fig_diag.savefig("model_diagnostics.png")
    
    # Sensitivity Analysis
    fig_sens = plot_sensitivity_analysis(models, df)
    fig_sens.savefig("sensitivity_analysis.png")
    
    print("\n✅ System Overhaul Complete.")
    print("👉 Launch the upgraded dashboard: streamlit run app.py")

if __name__ == "__main__":
    run_overhaul_pipeline()
