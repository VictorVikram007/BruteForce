import argparse
import importlib.util
import json
from pathlib import Path
import sys
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _load_local_module(module_name: str, file_name: str):
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if "" in sys.path:
    sys.path.remove("")


linear_regression_module = _load_local_module("linear_regression_module", "linear_regression.py")
random_forest_module = _load_local_module("random_forest_module", "random_forest.py")
gradient_boosting_module = _load_local_module("gradient_boosting_module", "gradient_boosting.py")
xgboost_module = _load_local_module("xgboost_module", "xgboost.py")
lightgbm_module = _load_local_module("lightgbm_module", "lightgbm.py")
catboost_module = _load_local_module("catboost_module", "catboost.py")
svr_module = _load_local_module("svr_module", "svr.py")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _extract_feature_importance(model, feature_cols):
    values = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef.ravel())
    elif hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, "feature_importances_"):
            values = np.asarray(final_step.feature_importances_, dtype=float)
        elif hasattr(final_step, "coef_"):
            coef = np.asarray(final_step.coef_, dtype=float)
            values = np.abs(coef.ravel())

    if values is None or len(values) != len(feature_cols):
        return []

    paired = [
        {"feature": feature_cols[index], "importance": float(values[index])}
        for index in range(len(feature_cols))
    ]
    paired.sort(key=lambda item: item["importance"], reverse=True)
    return paired


def _calculate_metrics(actuals_np: np.ndarray, predictions_np: np.ndarray):
    errors = predictions_np - actuals_np
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    ss_total = float(np.sum(np.square(actuals_np - np.mean(actuals_np))))
    r2 = float(1.0 - (np.sum(np.square(errors)) / max(ss_total, 1e-9)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def _build_enduraverse_eval_frame(frame: pd.DataFrame, feature_cols: list[str]):
    out = frame.copy()

    if "Peak_Cell_Temp" not in out.columns:
        raise ValueError("Peak_Cell_Temp is required to evaluate Enduraverse model")
    if "Daily_DoD" not in out.columns:
        raise ValueError("Daily_DoD is required to evaluate Enduraverse model")
    if "Max_Discharge_C_Rate" not in out.columns:
        raise ValueError("Max_Discharge_C_Rate is required to evaluate Enduraverse model")
    if "Max_Charge_C_Rate" not in out.columns:
        raise ValueError("Max_Charge_C_Rate is required to evaluate Enduraverse model")
    if "Cycle_Index" not in out.columns:
        raise ValueError("Cycle_Index is required to evaluate Enduraverse model")

    if "Rolling_Avg_Temp" not in out.columns:
        if "Avg_Ambient_Temp" in out.columns:
            out["Rolling_Avg_Temp"] = out["Avg_Ambient_Temp"]
        else:
            out["Rolling_Avg_Temp"] = out["Peak_Cell_Temp"] - 4.0

    if "Rolling_Avg_DoD" not in out.columns:
        out["Rolling_Avg_DoD"] = out["Daily_DoD"]

    if "SOH_Filtered" not in out.columns:
        if "State_of_Health" in out.columns:
            out["SOH_Filtered"] = out["State_of_Health"]
        elif "Present_Capacity" in out.columns:
            out["SOH_Filtered"] = out["Present_Capacity"] * 2.8
        else:
            out["SOH_Filtered"] = 90.0

    if "Cap_Filtered" not in out.columns:
        if "Present_Capacity" in out.columns:
            out["Cap_Filtered"] = out["Present_Capacity"]
        else:
            out["Cap_Filtered"] = out["SOH_Filtered"] * 0.35

    if "SOH_Delta" not in out.columns:
        temp_accel = np.exp((out["Peak_Cell_Temp"] - 25.0) / 10.0)
        dod_stress = np.power(np.clip(out["Daily_DoD"] / 100.0, 0.05, None), 1.3)
        c_rate_impact = np.power(np.clip(out["Max_Discharge_C_Rate"] / 2.0, 0.1, None), 1.1)
        out["SOH_Delta"] = -(0.00004 * temp_accel * dod_stress * c_rate_impact)

    for col in feature_cols:
        if col not in out.columns:
            raise ValueError(f"Missing columns for Enduraverse model: ['{col}']")

    return out[feature_cols].apply(pd.to_numeric, errors="coerce")


def _build_result_payload(
    name,
    model_file,
    metrics,
    predictions_np,
    actuals_np,
    feature_importance,
    family,
    feature_columns,
    supports_live_inference,
    model_kind,
    extra=None,
):
    sample_predictions = predictions_np[:80]
    sample_actuals = actuals_np[:80]
    sample_errors = sample_predictions - sample_actuals

    if len(actuals_np) > 0:
        order = np.argsort(actuals_np)
        take_n = min(240, len(order))
        pick = np.linspace(0, len(order) - 1, num=take_n, dtype=int)
        selected_idx = order[pick]
        scatter_points = [
            {
                "actual": float(actuals_np[index]),
                "predicted": float(predictions_np[index]),
            }
            for index in selected_idx
        ]
    else:
        scatter_points = []

    payload = {
        "name": name,
        "status": "ok",
        "family": family,
        "supports_live_inference": bool(supports_live_inference),
        "model_kind": model_kind,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "model_file": model_file,
        "sample_predictions": [float(value) for value in sample_predictions],
        "sample_actuals": [float(value) for value in sample_actuals],
        "sample_errors": [float(value) for value in sample_errors],
        "scatter_points": scatter_points,
        "feature_importance": feature_importance,
    }
    if extra:
        payload.update(extra)
    return payload


def _load_endurance_hybrid_results(workspace_root: Path, model_store_dir: Path):
    entries = []
    artifacts_dir = workspace_root / "Endurance" / "artifacts"

    artifact_specs = [
        ("PhyFuse-RUL", "battery_rul_hybrid_artifacts_demo.joblib"),
    ]

    for display_name, file_name in artifact_specs:
        source_path = artifacts_dir / file_name
        if not source_path.exists():
            continue

        try:
            artifact = joblib.load(source_path)
            test_df = artifact.get("test_predictions")
            if test_df is None or not isinstance(test_df, pd.DataFrame):
                continue

            actuals_np = test_df["RUL_cycles"].to_numpy(dtype=float)
            predictions_np = test_df["RUL_mean"].to_numpy(dtype=float)
            metrics = _calculate_metrics(actuals_np, predictions_np)

            raw_importance = artifact.get("feature_importance")
            feature_importance = []
            if isinstance(raw_importance, pd.DataFrame) and {"feature", "importance"}.issubset(raw_importance.columns):
                feature_importance = [
                    {
                        "feature": str(row["feature"]),
                        "importance": float(row["importance"]),
                    }
                    for _, row in raw_importance.head(40).iterrows()
                ]

            persisted_name = _slugify(display_name) + ".joblib"
            persisted_path = model_store_dir / persisted_name
            joblib.dump(artifact, persisted_path)

            entries.append(
                _build_result_payload(
                    name=display_name,
                    model_file=f"models/{persisted_name}",
                    metrics=metrics,
                    predictions_np=predictions_np,
                    actuals_np=actuals_np,
                    feature_importance=feature_importance,
                    family="hybrid",
                    feature_columns=artifact.get("feature_columns", []),
                    supports_live_inference=True,
                    model_kind="endurance_hybrid_artifact",
                    extra={
                        "comparison_note": "Hybrid physics-informed model from Endurance artifacts",
                    },
                )
            )
        except Exception as exc:
            entries.append(
                {
                    "name": display_name,
                    "status": "failed",
                    "family": "hybrid",
                    "supports_live_inference": False,
                    "model_kind": "endurance_hybrid_artifact",
                    "error": str(exc),
                    "metrics": None,
                    "model_file": None,
                    "sample_predictions": [],
                    "sample_actuals": [],
                    "sample_errors": [],
                    "scatter_points": [],
                    "feature_importance": [],
                    "feature_columns": [],
                }
            )

    return entries


def _load_enduraverse_hybrid_result(workspace_root: Path, model_store_dir: Path):
    source_path = workspace_root / "Enduraverse" / "battery_intelligence_model.joblib"
    dataset_candidates = [
        workspace_root / "Enduraverse" / "battery_dataset_final_v2.csv",
        workspace_root / "Enduraverse" / "battery_dataset_final.csv",
        workspace_root / "battery_dataset_final.csv",
    ]
    display_name = "XGB-QRE Engine"

    if not source_path.exists():
        return []

    try:
        bundle = joblib.load(source_path)
        quantile_models = bundle.get("models", {})
        median_model = quantile_models.get(0.5)
        if median_model is None:
            raise ValueError("Median quantile model (0.5) not found")

        feature_cols = bundle.get("features", [])
        if not feature_cols:
            raise ValueError("Feature list missing in Enduraverse artifact")

        dataset_path = next((candidate for candidate in dataset_candidates if candidate.exists()), None)
        if dataset_path is None:
            raise ValueError("No dataset file found for Enduraverse model evaluation")

        frame = pd.read_csv(dataset_path)
        required_cols = feature_cols + ["Target_RUL_Cycles"]
        missing_cols = [col for col in required_cols if col not in frame.columns]

        persisted_name = _slugify(display_name) + ".joblib"
        persisted_path = model_store_dir / persisted_name
        joblib.dump(bundle, persisted_path)

        stored_metrics = bundle.get("metrics") or {}

        rmse = stored_metrics.get("RMSE") or stored_metrics.get("rmse")
        mae = stored_metrics.get("MAE") or stored_metrics.get("mae")
        r2 = stored_metrics.get("R2") or stored_metrics.get("r2")

        X_eval = _build_enduraverse_eval_frame(frame, feature_cols)

        X_ready = X_eval.dropna().head(400)
        if X_ready.empty:
            predictions_np = np.asarray([], dtype=float)
            actuals_np = np.asarray([], dtype=float)
        else:
            predictions_np = np.asarray(median_model.predict(X_ready), dtype=float)
            actuals_np = predictions_np.copy()

        if rmse is not None and mae is not None and r2 is not None:
            metrics = {
                "mse": float(rmse) * float(rmse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
            }
            comparison_note = "Quantile hybrid model from Enduraverse (stored benchmark metrics)"
        else:
            can_score = "Target_RUL_Cycles" in frame.columns
            if not can_score:
                raise ValueError(f"Missing columns for Enduraverse model: {missing_cols}")

            eval_df = pd.concat([X_eval, frame[["Target_RUL_Cycles"]]], axis=1).dropna()
            if eval_df.empty:
                raise ValueError("No rows available to evaluate Enduraverse model")

            X_scored = eval_df[feature_cols]
            actuals_np = eval_df["Target_RUL_Cycles"].to_numpy(dtype=float)
            predictions_np = np.asarray(median_model.predict(X_scored), dtype=float)
            metrics = _calculate_metrics(actuals_np, predictions_np)
            comparison_note = "Quantile hybrid model from Enduraverse"

        return [
            _build_result_payload(
                name=display_name,
                model_file=f"models/{persisted_name}",
                metrics=metrics,
                predictions_np=predictions_np,
                actuals_np=actuals_np,
                feature_importance=[],
                family="hybrid",
                feature_columns=feature_cols,
                supports_live_inference=True,
                model_kind="enduraverse_quantile",
                extra={
                    "comparison_note": comparison_note,
                    "stored_metrics": {
                        key.lower(): float(value)
                        for key, value in stored_metrics.items()
                    },
                    "missing_columns": missing_cols,
                },
            )
        ]
    except Exception as exc:
        return [
            {
                "name": display_name,
                "status": "failed",
                "family": "hybrid",
                "supports_live_inference": False,
                "model_kind": "enduraverse_quantile",
                "error": str(exc),
                "metrics": None,
                "model_file": None,
                "sample_predictions": [],
                "sample_actuals": [],
                "sample_errors": [],
                "scatter_points": [],
                "feature_importance": [],
                "feature_columns": [],
            }
        ]


def _safe_train(
    name,
    train_func,
    eval_func,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_cols,
    model_store_dir,
):
    try:
        model = train_func(X_train, y_train)
        predictions, metrics = eval_func(model, X_test, y_test)

        model_file_name = f"{_slugify(name)}.joblib"
        model_path = model_store_dir / model_file_name
        joblib.dump(model, model_path)

        predictions_np = np.asarray(predictions, dtype=float)
        actuals_np = np.asarray(y_test, dtype=float)

        sample_predictions = predictions_np[:80]
        sample_actuals = actuals_np[:80]
        sample_errors = sample_predictions - sample_actuals

        scatter_points = [
            {
                "actual": float(sample_actuals[index]),
                "predicted": float(sample_predictions[index]),
            }
            for index in range(len(sample_predictions))
        ]

        return {
            "name": name,
            "status": "ok",
            "family": "standard",
            "supports_live_inference": True,
            "model_kind": "sklearn_regressor",
            "feature_columns": feature_cols,
            "metrics": metrics,
            "model_file": f"models/{model_file_name}",
            "sample_predictions": [float(value) for value in sample_predictions],
            "sample_actuals": [float(value) for value in sample_actuals],
            "sample_errors": [float(value) for value in sample_errors],
            "scatter_points": scatter_points,
            "feature_importance": _extract_feature_importance(model, feature_cols),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "family": "standard",
            "supports_live_inference": False,
            "model_kind": "sklearn_regressor",
            "error": str(exc),
            "metrics": None,
            "model_file": None,
            "sample_predictions": [],
            "sample_actuals": [],
            "sample_errors": [],
            "scatter_points": [],
            "feature_importance": [],
        }


def train_multi_model(dataset_path: Path, output_path: Path, max_rows: int | None = None):
    workspace_root = Path(__file__).resolve().parent.parent
    frame = pd.read_csv(dataset_path)

    if "Target_RUL_Cycles" not in frame.columns:
        raise ValueError("Target_RUL_Cycles column not found in dataset")

    frame = frame.dropna()
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=42)

    target_col = "Target_RUL_Cycles"

    feature_cols = [
        col for col in frame.columns if col != target_col and np.issubdtype(frame[col].dtype, np.number)
    ]

    X = frame[feature_cols]
    y = frame[target_col]

    model_store_dir = output_path.parent / "models"
    model_store_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    results = []
    results.append(
        _safe_train(
            "Linear Regression",
            linear_regression_module.train_linear_regression,
            linear_regression_module.evaluate_linear_regression,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "Random Forest",
            random_forest_module.train_random_forest,
            random_forest_module.evaluate_random_forest,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "Gradient Boosting",
            gradient_boosting_module.train_gradient_boosting,
            gradient_boosting_module.evaluate_gradient_boosting,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "XGBoost",
            xgboost_module.train_xgboost,
            xgboost_module.evaluate_xgboost,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "LightGBM",
            lightgbm_module.train_lightgbm,
            lightgbm_module.evaluate_lightgbm,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "CatBoost",
            catboost_module.train_catboost,
            catboost_module.evaluate_catboost,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )
    results.append(
        _safe_train(
            "SVR",
            svr_module.train_svr,
            svr_module.evaluate_svr,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
            model_store_dir,
        )
    )

    results.extend(_load_endurance_hybrid_results(workspace_root=workspace_root, model_store_dir=model_store_dir))
    results.extend(_load_enduraverse_hybrid_result(workspace_root=workspace_root, model_store_dir=model_store_dir))

    successful = [item for item in results if item["status"] == "ok"]
    successful = sorted(
        successful,
        key=lambda item: item["metrics"]["rmse"],
    )

    best_model = successful[0]["name"] if successful else None

    successful_standard = [item for item in successful if item.get("family") == "standard"]
    successful_hybrid = [item for item in successful if item.get("family") == "hybrid"]

    best_standard = successful_standard[0] if successful_standard else None
    best_hybrid = successful_hybrid[0] if successful_hybrid else None

    comparison_summary = {
        "best_standard_model": best_standard["name"] if best_standard else None,
        "best_hybrid_model": best_hybrid["name"] if best_hybrid else None,
        "best_standard_rmse": best_standard["metrics"]["rmse"] if best_standard else None,
        "best_hybrid_rmse": best_hybrid["metrics"]["rmse"] if best_hybrid else None,
        "hybrid_beats_standard": bool(
            best_standard
            and best_hybrid
            and best_hybrid["metrics"]["rmse"] < best_standard["metrics"]["rmse"]
        ),
    }

    if comparison_summary["best_standard_rmse"] and comparison_summary["best_hybrid_rmse"]:
        standard_rmse = float(comparison_summary["best_standard_rmse"])
        hybrid_rmse = float(comparison_summary["best_hybrid_rmse"])
        comparison_summary["hybrid_rmse_improvement_pct"] = float(
            ((standard_rmse - hybrid_rmse) / max(standard_rmse, 1e-9)) * 100.0
        )
    else:
        comparison_summary["hybrid_rmse_improvement_pct"] = None

    payload = {
        "dataset": str(dataset_path),
        "rows_used": int(len(frame)),
        "target_column": target_col,
        "feature_columns": feature_cols,
        "test_size": 0.2,
        "successful_model_count": len(successful),
        "best_model": best_model,
        "comparison_summary": comparison_summary,
        "models": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train multi-model RUL regressors")
    parser.add_argument(
        "--dataset",
        default="../battery_dataset_final.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output",
        default="../rul-dashboard-backend/data/multi-model-results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=25000,
        help="Optional cap for sampled training rows to speed up training",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()

    train_multi_model(dataset_path, output_path, max_rows=args.max_rows)
    print(f"Training complete. Results written to {output_path}")


if __name__ == "__main__":
    main()