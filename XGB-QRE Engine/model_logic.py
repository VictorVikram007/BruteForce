import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def split_data(df):
    """
    Splits data into train and test sets using GroupShuffleSplit to ensure
    that entire batteries are kept together (prevents data leakage).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["Battery_ID"]))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    feature_cols = [
        "SOH_Filtered", "Cap_Filtered", "SOH_Delta", 
        "Peak_Cell_Temp", "Daily_DoD", "Rolling_Avg_Temp", "Rolling_Avg_DoD",
        "Max_Discharge_C_Rate", "Max_Charge_C_Rate", "Cycle_Index"
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df["Target_RUL_Cycles"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["Target_RUL_Cycles"]
    
    return X_train, X_test, y_train, y_test, test_df["Battery_ID"]

def train_quantile_models(X_train, y_train):
    """
    Trains XGBoost models for three quantiles: 0.05, 0.5 (median), and 0.95.
    This provides robust probabilistic interval estimation.
    """
    quantiles = [0.05, 0.5, 0.95]
    models = {}
    
    for q in quantiles:
        print(f"Training XGBoost model for quantile {q}...")
        # XGBoost quantile regression setup
        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            tree_method='hist', # Efficient for large datasets
            random_state=42
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        models[q] = pipeline
        
    return models

def predict_quantiles(models, X):
    """
    Generates predictions for all three quantiles.
    """
    preds = {}
    for q, pipeline in models.items():
        preds[q] = pipeline.predict(X)
    return preds

def evaluate_models(y_test, preds):
    """
    Evaluates the median prediction and the quality of the quantile intervals.
    """
    y_pred_median = preds[0.5]
    
    mae = mean_absolute_error(y_test, y_pred_median)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_median))
    r2 = r2_score(y_test, y_pred_median)
    
    # Coverage probability (Fraction of points within the 90% interval)
    coverage = np.mean((y_test >= preds[0.05]) & (y_test <= preds[0.95]))
    
    print(f"\n--- Model Overhaul Diagnostics ---")
    print(f"MAE: {mae:.2f} cycles")
    print(f"RMSE: {rmse:.2f} cycles")
    print(f"R2 Score: {r2:.4f}")
    print(f"90% Prediction Interval Coverage: {coverage*100:.1f}%")
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Coverage": coverage}

if __name__ == "__main__":
    df = pd.read_csv("battery_dataset_final.csv")
    X_train, X_test, y_train, y_test, groups = split_data(df)
    
    models = train_quantile_models(X_train, y_train)
    preds = predict_quantiles(models, X_test)
    metrics = evaluate_models(y_test, preds)
    
    # Save the ensemble
    joblib.dump({
        'models': models,
        'features': X_train.columns.tolist(),
        'metrics': metrics
    }, "battery_intelligence_model.joblib")
    print("\n✅ Upgraded Model saved to battery_intelligence_model.joblib")
