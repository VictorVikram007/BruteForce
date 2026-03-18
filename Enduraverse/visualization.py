import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def plot_degradation_trends(df, selected_ids=None):
    """
    Plots filtered SOH and Capacity trends for selected batteries.
    Highlights the 'High Stress Case' for storytelling.
    """
    if selected_ids is None:
        selected_ids = df["Battery_ID"].unique()[:3]
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for b_id in selected_ids:
        b_df = df[df["Battery_ID"] == b_id]
        
        # Elite Storytelling: Highlight Battery 0
        lw = 3.5 if b_id == selected_ids[0] else 2.0
        alpha = 1.0 if b_id == selected_ids[0] else 0.7
        label = f"Battery {b_id} (High Stress)" if b_id == selected_ids[0] else f"Battery {b_id}"
        
        # Use Filtered values for smoother plots
        sns.lineplot(ax=axes[0], data=b_df, x="Cycle_Index", y="SOH_Filtered", lw=lw, alpha=alpha, label=label)
        sns.lineplot(ax=axes[1], data=b_df, x="Cycle_Index", y="Cap_Filtered", lw=lw, alpha=alpha, label=label)
        
        # Mark EOL
        eol_data = b_df[b_df["State_of_Health"] <= 80].head(1)
        if not eol_data.empty:
            axes[0].scatter(eol_data["Cycle_Index"], eol_data["SOH_Filtered"], color='red', marker='o', s=60, zorder=5)
            
    axes[0].set_title("SOH Trends (Kalman Filtered)", fontweight='bold')
    axes[0].axhline(80, ls='--', color='red', alpha=0.6)
    axes[0].text(df["Cycle_Index"].max()*0.05, 81, "End-of-Life (80%)", color='red', fontweight='bold', alpha=0.8)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].set_title("Capacity Trends (Filtered)", fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_diagnostic_dashboard(y_test, preds, metrics):
    """
    Advanced Diagnostic Dashboard for ML Overhaul.
    Includes:
    1. Predicted vs Actual (Scatter with Error Density)
    2. Residual Plot
    3. Error Distribution
    4. Calibration Curve (Quantile Coverage)
    """
    y_median = preds[0.5]
    residuals = y_test - y_median
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Predicted vs Actual (Color-coded by error)
    error_mag = np.abs(residuals)
    sc = axes[0, 0].scatter(y_test, y_median, c=error_mag, cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel("Actual RUL (Cycles)")
    axes[0, 0].set_ylabel("Predicted RUL (Cycles)")
    axes[0, 0].set_title(f"Predicted vs Actual (R²={metrics['R2']:.3f})", fontweight='bold')
    plt.colorbar(sc, ax=axes[0, 0], label="Absolute Error (Cycles)")
    
    # 2. Residual Plot
    axes[0, 1].scatter(y_median, residuals, alpha=0.5, s=20, color='teal')
    axes[0, 1].axhline(0, color='red', ls='--')
    axes[0, 1].set_xlabel("Predicted RUL (Cycles)")
    axes[0, 1].set_ylabel("Residuals (Actual - Predicted)")
    axes[0, 1].set_title("Residual Analysis (Error Patterns)", fontweight='bold')
    
    # 3. Error Distribution (Histogram)
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='purple')
    axes[1, 0].set_xlabel("Error (Cycles)")
    axes[1, 0].set_title("Error Distribution (Normalcy Check)", fontweight='bold')
    
    # 4. Calibration Curve (Simplified: Theoretical vs Actual Coverage)
    # We check if roughly 45% of data is above 0.5 and 45% below, etc.
    # More simply, we plot the predicted width vs error
    width = preds[0.95] - preds[0.05]
    axes[1, 1].scatter(width, error_mag, alpha=0.4, s=20, color='orange')
    axes[1, 1].set_xlabel("Predicted 90% Interval Width (Uncertainty)")
    axes[1, 1].set_ylabel("Actual Prediction Error")
    axes[1, 1].set_title("Calibration: Uncertainty vs. Error Magnitude", fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_sensitivity_analysis(models, df):
    """
    Shows impact of Temp, DoD, and C-rate using the median model.
    """
    pipeline = models[0.5]
    feature_cols = [
        "SOH_Filtered", "Cap_Filtered", "SOH_Delta", 
        "Peak_Cell_Temp", "Daily_DoD", "Rolling_Avg_Temp", "Rolling_Avg_DoD",
        "Max_Discharge_C_Rate", "Max_Charge_C_Rate", "Cycle_Index"
    ]
    
    # Baseline: Median values from dataset
    baseline = df[feature_cols].median().to_frame().T
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sensitivity to Temp
    temps = np.linspace(20, 60, 50)
    temp_preds = []
    for t in temps:
        sample = baseline.copy()
        sample["Peak_Cell_Temp"] = t
        sample["Rolling_Avg_Temp"] = t
        temp_preds.append(pipeline.predict(sample)[0])
    axes[0].plot(temps, temp_preds, lw=2, color='red')
    axes[0].set_title("Sensitivity: Temperature")
    axes[0].set_xlabel("Peak Cell Temp (°C)")
    axes[0].set_ylabel("Predicted RUL")
    
    # Sensitivity to DoD
    dods = np.linspace(40, 100, 50)
    dod_preds = []
    for d in dods:
        sample = baseline.copy()
        sample["Daily_DoD"] = d
        sample["Rolling_Avg_DoD"] = d
        dod_preds.append(pipeline.predict(sample)[0])
    axes[1].plot(dods, dod_preds, lw=2, color='blue')
    axes[1].set_title("Sensitivity: Depth of Discharge")
    axes[1].set_xlabel("Daily DoD (%)")
    
    # Sensitivity to C-Rate
    rates = np.linspace(0.5, 4.0, 50)
    rate_preds = []
    for r in rates:
        sample = baseline.copy()
        sample["Max_Discharge_C_Rate"] = r
        rate_preds.append(pipeline.predict(sample)[0])
    axes[2].plot(rates, rate_preds, lw=2, color='green')
    axes[2].set_title("Sensitivity: C-Rate")
    axes[2].set_xlabel("Discharge C-Rate")
    
    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    return fig
