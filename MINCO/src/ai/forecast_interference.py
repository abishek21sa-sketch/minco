from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
AI_MODELS_DIR = RESULTS_DIR / "ai_models"
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"

MODEL_REGISTRY_PATH = RESULTS_DIR / "ai_reports" / "forecast_model_registry.csv"


# ============================================================
# IO
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Registry / model lookup
# ============================================================

def load_model_registry() -> pd.DataFrame:
    df = safe_read_csv(MODEL_REGISTRY_PATH)
    if df.empty:
        raise FileNotFoundError(
            f"Model registry not found or empty: {MODEL_REGISTRY_PATH}"
        )
    return df


def choose_best_model_name(task_name: str) -> str:
    """
    Simple default selection based on training results we observed:
    - blocked arrivals: random forest regressor
    - unsafe excess: linear regression often comparable / slightly better
    - critical utilization: random forest classifier
    - next regime: logistic regression is sufficient here
    """
    preferred = {
        "forecast_next_unsafe_excess": "linear_regression",
        "forecast_next_blocked_arrivals": "random_forest_regressor",
        "forecast_next_utilization_critical": "random_forest_classifier",
        "forecast_next_regime": "logistic_regression",
    }
    return preferred[task_name]


def get_model_path(
    task_name: str,
    model_name: str | None = None,
    registry_df: pd.DataFrame | None = None,
) -> Path:
    if registry_df is None:
        registry_df = load_model_registry()

    if model_name is None:
        model_name = choose_best_model_name(task_name)

    rows = registry_df[
        (registry_df["task_name"] == task_name)
        & (registry_df["model_name"] == model_name)
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No model found in registry for task='{task_name}', model='{model_name}'"
        )

    return Path(rows.iloc[0]["model_path"])


# ============================================================
# Inference feature alignment
# ============================================================

def add_missing_context_defaults(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    defaults = {
        "time_index": 0,
        "current_regime_code": 0,
        "lookahead_horizon": 7,
        "icu_capacity_scale": 1.0,
        "transfer_capacity_scale": 1.0,
        "demand_scale": 1.0,

        "total_unsafe_excess": 0.0,
        "total_blocked_arrivals": 0.0,
        "max_utilization_ratio": 1.0,
        "num_unsafe_rows": 0.0,
        "total_overflow_excess": 0.0,
        "total_surge_gap": 0.0,

        "total_unsafe_excess_lag1": 0.0,
        "total_unsafe_excess_lag2": 0.0,
        "total_unsafe_excess_rollmean_3": 0.0,
        "total_unsafe_excess_rollmax_3": 0.0,

        "total_blocked_arrivals_lag1": 0.0,
        "total_blocked_arrivals_lag2": 0.0,
        "total_blocked_arrivals_rollmean_3": 0.0,
        "total_blocked_arrivals_rollmax_3": 0.0,

        "max_utilization_ratio_lag1": 1.0,
        "max_utilization_ratio_lag2": 1.0,
        "max_utilization_ratio_rollmean_3": 1.0,
        "max_utilization_ratio_rollmax_3": 1.0,

        "num_unsafe_rows_lag1": 0.0,
        "num_unsafe_rows_lag2": 0.0,
        "num_unsafe_rows_rollmean_3": 0.0,
        "num_unsafe_rows_rollmax_3": 0.0,

        "total_overflow_excess_lag1": 0.0,
        "total_overflow_excess_lag2": 0.0,
        "total_overflow_excess_rollmean_3": 0.0,
        "total_overflow_excess_rollmax_3": 0.0,

        "total_surge_gap_lag1": 0.0,
        "total_surge_gap_lag2": 0.0,
        "total_surge_gap_rollmean_3": 0.0,
        "total_surge_gap_rollmax_3": 0.0,

        "unsafe_positive_flag": 0,
        "unsafe_high_flag": 0,
        "blocked_positive_flag": 0,
        "blocked_high_flag": 0,
        "utilization_high_flag": 0,
        "utilization_critical_flag": 0,
        "overflow_positive_flag": 0,
        "unsafe_rows_positive_flag": 0,

        "policy_name": "optimized_network",
        "scenario": "baseline",
        "design_name": "live_inference",
        "dataset_source": "live_inference",
        "current_regime": "normal",
    }

    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val

    return out


def regime_code_from_name(regime: str) -> int:
    mapping = {"normal": 0, "surge": 1, "crisis": 2}
    return mapping.get(str(regime).strip().lower(), 0)


def prepare_inference_frame(
    state_row: dict[str, Any] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(state_row, pd.DataFrame):
        df = state_row.copy()
    elif isinstance(state_row, pd.Series):
        df = pd.DataFrame([state_row.to_dict()])
    elif isinstance(state_row, dict):
        df = pd.DataFrame([state_row])
    else:
        raise TypeError(
            f"Unsupported state_row type: {type(state_row)}"
        )

    df = add_missing_context_defaults(df)

    if "current_regime" in df.columns and "current_regime_code" in df.columns:
        df["current_regime"] = df["current_regime"].astype(str).str.strip().str.lower()
        df["current_regime_code"] = df["current_regime"].apply(regime_code_from_name)

    # numeric coercion for known numeric columns
    numeric_like_cols = [
        c for c in df.columns
        if (
            c.endswith("_lag1")
            or c.endswith("_lag2")
            or c.endswith("_rollmean_3")
            or c.endswith("_rollmax_3")
            or c in {
                "time_index",
                "current_regime_code",
                "lookahead_horizon",
                "icu_capacity_scale",
                "transfer_capacity_scale",
                "demand_scale",
                "total_unsafe_excess",
                "total_blocked_arrivals",
                "max_utilization_ratio",
                "num_unsafe_rows",
                "total_overflow_excess",
                "total_surge_gap",
                "unsafe_positive_flag",
                "unsafe_high_flag",
                "blocked_positive_flag",
                "blocked_high_flag",
                "utilization_high_flag",
                "utilization_critical_flag",
                "overflow_positive_flag",
                "unsafe_rows_positive_flag",
            }
        )
    ]

    for col in numeric_like_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)
    return df


# ============================================================
# Core inference
# ============================================================

def predict_with_model(
    task_name: str,
    input_df: pd.DataFrame,
    model_name: str | None = None,
    registry_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if registry_df is None:
        registry_df = load_model_registry()

    model_path = get_model_path(
        task_name=task_name,
        model_name=model_name,
        registry_df=registry_df,
    )
    model = load_pickle(model_path)

    preds = model.predict(input_df)

    result = {
        "task_name": task_name,
        "model_name": model_name if model_name is not None else choose_best_model_name(task_name),
        "model_path": str(model_path),
        "prediction": preds.tolist(),
    }

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(input_df)
            result["prediction_proba"] = probs.tolist()

            classes = None
            try:
                classes = model.named_steps["model"].classes_
            except Exception:
                pass

            if classes is not None:
                result["classes"] = [str(c) for c in classes.tolist()]
        except Exception:
            pass

    return result


def forecast_all(
    state_row: dict[str, Any] | pd.Series | pd.DataFrame,
) -> dict[str, Any]:
    input_df = prepare_inference_frame(state_row)
    registry_df = load_model_registry()

    unsafe_res = predict_with_model(
        task_name="forecast_next_unsafe_excess",
        input_df=input_df,
        registry_df=registry_df,
    )
    blocked_res = predict_with_model(
        task_name="forecast_next_blocked_arrivals",
        input_df=input_df,
        registry_df=registry_df,
    )
    util_res = predict_with_model(
        task_name="forecast_next_utilization_critical",
        input_df=input_df,
        registry_df=registry_df,
    )
    regime_res = predict_with_model(
        task_name="forecast_next_regime",
        input_df=input_df,
        registry_df=registry_df,
    )

    out = {
        "n_rows": len(input_df),
        "predicted_next_unsafe_excess": unsafe_res["prediction"],
        "predicted_next_blocked_arrivals": blocked_res["prediction"],
        "predicted_next_utilization_critical_flag": util_res["prediction"],
        "predicted_next_regime_label": regime_res["prediction"],
        "unsafe_model": unsafe_res["model_name"],
        "blocked_model": blocked_res["model_name"],
        "utilization_model": util_res["model_name"],
        "regime_model": regime_res["model_name"],
    }

    if "prediction_proba" in util_res:
        out["predicted_next_utilization_critical_proba"] = util_res["prediction_proba"]
        if "classes" in util_res:
            out["predicted_next_utilization_classes"] = util_res["classes"]

    if "prediction_proba" in regime_res:
        out["predicted_next_regime_proba"] = regime_res["prediction_proba"]
        if "classes" in regime_res:
            out["predicted_next_regime_classes"] = regime_res["classes"]

    return out


# ============================================================
# Convenience wrappers
# ============================================================

def forecast_one(state_row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    out = forecast_all(state_row)

    return {
        "predicted_next_unsafe_excess": float(out["predicted_next_unsafe_excess"][0]),
        "predicted_next_blocked_arrivals": float(out["predicted_next_blocked_arrivals"][0]),
        "predicted_next_utilization_critical_flag": str(out["predicted_next_utilization_critical_flag"][0]),
        "predicted_next_regime_label": str(out["predicted_next_regime_label"][0]),
        "predicted_next_utilization_critical_proba": (
            out["predicted_next_utilization_critical_proba"][0]
            if "predicted_next_utilization_critical_proba" in out else None
        ),
        "predicted_next_regime_proba": (
            out["predicted_next_regime_proba"][0]
            if "predicted_next_regime_proba" in out else None
        ),
        "unsafe_model": out["unsafe_model"],
        "blocked_model": out["blocked_model"],
        "utilization_model": out["utilization_model"],
        "regime_model": out["regime_model"],
    }


def pretty_print_forecast(forecast: dict[str, Any]) -> None:
    print("\nFORECAST INFERENCE")
    print("=" * 60)
    for k, v in forecast.items():
        print(f"{k}: {v}")


# ============================================================
# Example main
# ============================================================

def main() -> None:
    example_state = {
        "policy_name": "regime_robust_optimized_network",
        "scenario": "baseline",
        "design_name": "live_inference",
        "dataset_source": "live_inference",
        "current_regime": "surge",

        "time_index": 10,
        "lookahead_horizon": 7,
        "icu_capacity_scale": 1.0,
        "transfer_capacity_scale": 1.0,
        "demand_scale": 1.0,

        "total_unsafe_excess": 4.1,
        "total_blocked_arrivals": 14.5,
        "max_utilization_ratio": 1.019,
        "num_unsafe_rows": 1.87,
        "total_overflow_excess": 1.58,
        "total_surge_gap": 0.44,

        "total_unsafe_excess_lag1": 4.5,
        "total_unsafe_excess_lag2": 3.8,
        "total_unsafe_excess_rollmean_3": 4.0,
        "total_unsafe_excess_rollmax_3": 4.8,

        "total_blocked_arrivals_lag1": 13.0,
        "total_blocked_arrivals_lag2": 12.6,
        "total_blocked_arrivals_rollmean_3": 13.4,
        "total_blocked_arrivals_rollmax_3": 14.5,

        "max_utilization_ratio_lag1": 1.05,
        "max_utilization_ratio_lag2": 1.02,
        "max_utilization_ratio_rollmean_3": 1.03,
        "max_utilization_ratio_rollmax_3": 1.08,

        "num_unsafe_rows_lag1": 2.0,
        "num_unsafe_rows_lag2": 1.0,
        "num_unsafe_rows_rollmean_3": 1.63,
        "num_unsafe_rows_rollmax_3": 2.0,

        "unsafe_positive_flag": 1,
        "unsafe_high_flag": 0,
        "blocked_positive_flag": 1,
        "blocked_high_flag": 1,
        "utilization_high_flag": 0,
        "utilization_critical_flag": 0,
        "overflow_positive_flag": 1,
        "unsafe_rows_positive_flag": 1,
    }

    forecast = forecast_one(example_state)
    pretty_print_forecast(forecast)


if __name__ == "__main__":
    main()