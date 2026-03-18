import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = str(Path(__file__).resolve().parent)
if CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if "" in sys.path:
    sys.path.remove("")


def _as_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _map_enduraverse_features(incoming: dict, feature_columns: list[str]):
    peak_temp = _as_float(incoming.get("Peak_Cell_Temp"), incoming.get("peak_cell_temp", 35.0))
    daily_dod = _as_float(incoming.get("Daily_DoD"), incoming.get("daily_dod", 75.0))
    rolling_avg_temp = _as_float(
        incoming.get("Rolling_Avg_Temp"), incoming.get("Avg_Ambient_Temp", 32.0)
    )
    rolling_avg_dod = _as_float(incoming.get("Rolling_Avg_DoD"), daily_dod)
    max_discharge = _as_float(incoming.get("Max_Discharge_C_Rate"), 1.5)
    max_charge = _as_float(incoming.get("Max_Charge_C_Rate"), 1.0)
    cycle_index = _as_float(incoming.get("Cycle_Index"), incoming.get("cycle_index", 900.0))

    state_of_health = _as_float(incoming.get("State_of_Health"), 90.0)
    present_capacity = _as_float(incoming.get("Present_Capacity"), 31.5)

    if state_of_health <= 0.0 and present_capacity > 0.0:
        state_of_health = max(60.0, min(100.0, present_capacity * 2.8))

    stress_base = 0.00004
    temp_accel = 2.718281828 ** ((peak_temp - 25.0) / 10.0)
    dod_stress = max(daily_dod / 100.0, 0.05) ** 1.3
    c_rate_impact = max(max_discharge / 2.0, 0.1) ** 1.1
    soh_delta = -(stress_base * temp_accel * dod_stress * c_rate_impact)

    mapped = {
        "SOH_Filtered": state_of_health,
        "Cap_Filtered": present_capacity,
        "SOH_Delta": soh_delta,
        "Peak_Cell_Temp": peak_temp,
        "Daily_DoD": daily_dod,
        "Rolling_Avg_Temp": rolling_avg_temp,
        "Rolling_Avg_DoD": rolling_avg_dod,
        "Max_Discharge_C_Rate": max_discharge,
        "Max_Charge_C_Rate": max_charge,
        "Cycle_Index": cycle_index,
    }

    row = {}
    for column in feature_columns:
        if column in incoming:
            row[column] = _as_float(incoming[column])
        elif column in mapped:
            row[column] = float(mapped[column])
        else:
            raise ValueError(f"Missing required feature: {column}")

    return row


def _map_endurance_features(incoming: dict, artifacts: dict):
    feature_columns = artifacts.get("feature_columns", [])
    defaults = artifacts.get("feature_medians", {}) if isinstance(artifacts.get("feature_medians"), dict) else {}

    row = {column: float(defaults.get(column, 0.0)) for column in feature_columns}

    cycle_index = _as_float(incoming.get("Cycle_Index"), _as_float(incoming.get("cycle_index"), row.get("cycle_index", 900.0)))
    peak_temp = _as_float(incoming.get("Peak_Cell_Temp"), row.get("max_temp", 40.0))
    ambient_temp = _as_float(incoming.get("Avg_Ambient_Temp"), row.get("avg_temp", max(peak_temp - 6.0, 20.0)))
    daily_dod_pct = _as_float(incoming.get("Daily_DoD"), row.get("dod", 0.6) * 100.0)
    dod = np.clip(daily_dod_pct / 100.0, 0.05, 0.95)
    discharge_c_rate = _as_float(incoming.get("Max_Discharge_C_Rate"), row.get("c_rate", 1.2))
    charge_c_rate = _as_float(incoming.get("Max_Charge_C_Rate"), max(discharge_c_rate * 0.6, 0.5))
    rolling_temp = _as_float(incoming.get("Rolling_Avg_Temp"), ambient_temp)
    rolling_dod = _as_float(incoming.get("Rolling_Avg_DoD"), daily_dod_pct) / 100.0
    daily_km = _as_float(incoming.get("Daily_km"), _as_float(incoming.get("dailyKm"), 135.0))

    row["cycle_index"] = cycle_index
    row["calendar_age_days"] = cycle_index
    row["avg_temp"] = np.clip(ambient_temp, 20.0, 45.0)
    row["max_temp"] = np.clip(peak_temp, row["avg_temp"], 55.0)
    row["temp_gradient"] = max(row["max_temp"] - row["avg_temp"], 0.0)
    row["dod"] = float(dod)
    row["dod_rolling_avg"] = float(np.clip(rolling_dod, 0.05, 0.95))
    row["dod_variance"] = float(max((row["dod"] - row["dod_rolling_avg"]) ** 2, 0.0))
    row["c_rate"] = float(np.clip(discharge_c_rate, 0.2, 3.0))
    row["fast_charge_ratio"] = 1.0 if charge_c_rate > 1.0 else 0.0
    row["avg_charging_time"] = max(2200.0, 9800.0 - 2400.0 * row["fast_charge_ratio"])
    row["avg_speed"] = float(np.clip(daily_km / 5.2, 12.0, 45.0))
    row["stop_go_ratio"] = float(np.clip(0.35 + 0.4 * (1.0 - row["avg_speed"] / 45.0), 0.2, 0.9))
    row["acceleration_events"] = float(round(row["stop_go_ratio"] * 80 + row["dod"] * 25))
    row["regen_braking_events"] = float(round(row["acceleration_events"] * 0.65))
    row["rolling_avg_temp"] = rolling_temp
    row["rolling_avg_dod"] = rolling_dod

    row["cycle_stress"] = float(np.power(row["dod"], 1.3) * np.power(max(row["c_rate"], 0.1), 1.1))
    row["calendar_stress"] = float(np.exp((row["avg_temp"] - 25.0) / 10.0) * (row["calendar_age_days"] / 365.0))
    row["total_degradation"] = float(row["cycle_stress"] + 0.3 * row["calendar_stress"])
    row["degradation_score"] = float(np.exp(row["avg_temp"] / 40.0) * np.power(row["dod"], 1.3) * np.power(max(row["c_rate"], 0.1), 1.1))
    row["thermal_stress"] = float(max(row["avg_temp"] - 40.0, 0.0) * row["avg_temp"] * 300.0)
    row["degradation_rate"] = float(max(0.00035 * row["degradation_score"], 1e-5))
    row["high_temp_flag"] = 1 if row["avg_temp"] >= 40.0 else 0
    row["high_dod_flag"] = 1 if row["dod"] >= 0.80 else 0

    if "midc_energy_per_cycle" in row:
        row["midc_energy_per_cycle"] = float(0.25 * (1.0 + 0.45 * row["stop_go_ratio"] + 0.002 * row["acceleration_events"]))
    if "midc_stress_score" in row:
        row["midc_stress_score"] = float(row["midc_energy_per_cycle"] * (1.0 + 0.03 * max(row["avg_temp"] - 25.0, 0.0)))
    if "time_above_40C" in row:
        row["time_above_40C"] = float((max(row["avg_temp"] - 40.0, 0.0) / 5.0) * 4200.0)

    for key in row:
        row[key] = float(row[key])

    return row


def predict(registry_path: Path, model_name: str, features_json: str):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    selected = None
    for model in registry.get("models", []):
        if model.get("name", "").lower() == model_name.lower():
            selected = model
            break

    if not selected:
        raise ValueError(f"Model '{model_name}' was not found")

    if selected.get("status") != "ok":
        raise ValueError(f"Model '{selected.get('name')}' is not available for inference")

    model_file = selected.get("model_file")
    if not model_file:
        raise ValueError("Model file path is missing in registry")

    model_path = registry_path.parent / model_file
    if not model_path.exists():
        raise ValueError(f"Persisted model not found at {model_path}")

    feature_columns = selected.get("feature_columns") or registry.get("feature_columns", [])
    incoming = json.loads(features_json)

    if not selected.get("supports_live_inference", True):
        raise ValueError(
            "Live inference is not supported for this model in the dashboard. "
            "Use Model Comparison to evaluate this model."
        )

    model_kind = selected.get("model_kind", "sklearn_regressor")

    if model_kind == "enduraverse_quantile":
        row = _map_enduraverse_features(incoming, feature_columns)
    elif model_kind == "endurance_hybrid_artifact":
        artifacts = joblib.load(model_path)
        row = _map_endurance_features(incoming, artifacts)
        feature_columns = artifacts.get("feature_columns", feature_columns)
    else:
        row = {}
        for column in feature_columns:
            if column not in incoming:
                raise ValueError(f"Missing required feature: {column}")
            row[column] = float(incoming[column])

    frame = pd.DataFrame([row], columns=feature_columns)

    if model_kind == "enduraverse_quantile":
        bundle = joblib.load(model_path)
        quantile_models = bundle.get("models", {})
        median_model = quantile_models.get(0.5)
        lower_model = quantile_models.get(0.05)
        upper_model = quantile_models.get(0.95)
        if median_model is None:
            raise ValueError("Enduraverse median quantile model is unavailable")

        prediction = float(median_model.predict(frame)[0])
        payload = {
            "model": selected.get("name"),
            "prediction": prediction,
            "features_used": row,
        }

        if lower_model is not None and upper_model is not None:
            payload["prediction_lower"] = float(lower_model.predict(frame)[0])
            payload["prediction_upper"] = float(upper_model.predict(frame)[0])

        print(json.dumps(payload))
        return

    if model_kind == "endurance_hybrid_artifact":
        artifacts = joblib.load(model_path)
        x = frame.reindex(columns=artifacts["feature_columns"]).copy()
        x = x.apply(pd.to_numeric, errors="coerce")

        fill_values = artifacts.get("feature_medians", {})
        if isinstance(fill_values, dict) and fill_values:
            x = x.fillna(fill_values)
        x = x.fillna(0.0)

        blend_weights = artifacts.get("blend_weights", {"direct": 0.5, "hybrid": 0.5})
        w_direct = float(blend_weights.get("direct", 0.5))
        w_hybrid = float(blend_weights.get("hybrid", 0.5))
        w_sum = max(w_direct + w_hybrid, 1e-6)
        w_direct /= w_sum
        w_hybrid /= w_sum

        ev_cfg = artifacts.get("ev_config", {})
        eol_soh_pct = float(ev_cfg.get("eol_soh_pct", 80.0))
        soh_min_frac = eol_soh_pct / 100.0

        pred_direct = np.asarray(artifacts["direct_model"].predict(x), dtype=float)
        pred_soh = np.clip(np.asarray(artifacts["soh_model"].predict(x), dtype=float), 0.0, 100.0)
        soh_frac = np.clip(pred_soh / 100.0, soh_min_frac + 1e-6, 1.05)
        degradation_rate = np.clip(x["degradation_rate"].to_numpy(dtype=float), 1e-6, None)
        total_degradation = np.clip(x["total_degradation"].to_numpy(dtype=float), 0.0, None)
        k = 0.00008 + 0.12 * degradation_rate + 0.00025 * total_degradation
        k = np.clip(k, 1e-5, 0.02)
        pred_hybrid = np.clip((1.0 / k) * np.log(soh_frac / soh_min_frac), 0.0, None)

        raw_pred = (w_direct * pred_direct) + (w_hybrid * pred_hybrid)
        cal_model = artifacts.get("calibration_model")
        if cal_model is not None:
            mean_pred = np.asarray(cal_model.predict(raw_pred.reshape(-1, 1)), dtype=float)
        else:
            mean_pred = raw_pred

        lower = np.asarray(artifacts["q_lower_model"].predict(x), dtype=float)
        upper = np.asarray(artifacts["q_upper_model"].predict(x), dtype=float)
        confidence = np.clip(1.0 - (upper - lower) / np.clip(mean_pred, 1.0, None), 0.0, 1.0)

        payload = {
            "model": selected.get("name"),
            "prediction": float(mean_pred[0]),
            "prediction_lower": float(lower[0]),
            "prediction_upper": float(upper[0]),
            "confidence_score": float(confidence[0]),
            "features_used": row,
        }
        print(json.dumps(payload))
        return

    model = joblib.load(model_path)
    prediction = float(model.predict(frame)[0])

    payload = {
        "model": selected.get("name"),
        "prediction": prediction,
        "features_used": row,
    }
    print(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser(description="Predict RUL from persisted model")
    parser.add_argument("--registry", required=True, help="Path to multi-model-results.json")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--features-json", required=True, help="JSON string with feature values")

    args = parser.parse_args()

    predict(
        registry_path=Path(args.registry).resolve(),
        model_name=args.model,
        features_json=args.features_json,
    )


if __name__ == "__main__":
    main()