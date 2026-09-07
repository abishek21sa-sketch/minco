from __future__ import annotations

from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"
TABLES_DIR = RESULTS_DIR / "tables"
SYSTEM_DIR = RESULTS_DIR / "system_pipeline"
SYSTEM_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_DATA_PATH = AI_DATA_DIR / "forecast_dataset_combined.csv"

FORECAST_REGISTRY_PATH = AI_REPORTS_DIR / "forecast_model_registry_leakage_safe.csv"
FORECAST_RESULTS_PATH = AI_REPORTS_DIR / "forecast_model_results_leakage_safe.csv"
STRICT_REGIME_RESULTS_PATH = AI_REPORTS_DIR / "strict_balanced_regime_results.csv"
UNSAFE_RISK_RESULTS_PATH = AI_REPORTS_DIR / "unsafe_excess_risk_classifier_results.csv"

REGIME_SUMMARY_PATH = TABLES_DIR / "regime_suite_summary.csv"

OUTPUT_JSON_PATH = SYSTEM_DIR / "full_pipeline_decision_report.json"
OUTPUT_SCORE_PATH = SYSTEM_DIR / "full_pipeline_policy_scores.csv"
OUTPUT_TRACE_PATH = SYSTEM_DIR / "full_pipeline_trace.csv"
OUTPUT_PREDICTION_PATH = SYSTEM_DIR / "full_pipeline_predictions.csv"


DEFAULT_SCENARIO = "baseline"

POLICY_CANDIDATES = [
    "optimized_network",
    "robust_optimized_network",
    "regime_robust_optimized_network",
]

DEFAULT_POLICY_WEIGHTS = {
    "unsafe_weight": 1.00,
    "blocked_weight": 0.35,
    "utilization_weight": 2.00,
    "risk_weight": 4.00,
    "regime_crisis_weight": 1.50,
    "forecast_blocked_weight": 0.15,
    "forecast_unsafe_weight": 0.10,
    "forecast_utilization_risk_weight": 2.00,
}


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def add_pipeline_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "time_index",
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
        "total_unsafe_excess_lag1",
        "total_unsafe_excess_lag2",
        "total_unsafe_excess_rollmean_3",
        "total_unsafe_excess_rollmax_3",
        "total_blocked_arrivals_lag1",
        "total_blocked_arrivals_lag2",
        "total_blocked_arrivals_rollmean_3",
        "total_blocked_arrivals_rollmax_3",
        "max_utilization_ratio_lag1",
        "max_utilization_ratio_lag2",
        "max_utilization_ratio_rollmean_3",
        "max_utilization_ratio_rollmax_3",
        "num_unsafe_rows_lag1",
        "num_unsafe_rows_lag2",
        "num_unsafe_rows_rollmean_3",
        "num_unsafe_rows_rollmax_3",
        "total_overflow_excess_lag1",
        "total_overflow_excess_lag2",
        "total_overflow_excess_rollmean_3",
        "total_overflow_excess_rollmax_3",
        "total_surge_gap_lag1",
        "total_surge_gap_lag2",
        "total_surge_gap_rollmean_3",
        "total_surge_gap_rollmax_3",
        "unsafe_positive_flag",
        "unsafe_high_flag",
        "blocked_positive_flag",
        "blocked_high_flag",
        "utilization_high_flag",
        "utilization_critical_flag",
        "overflow_positive_flag",
        "unsafe_rows_positive_flag",
    ]

    categorical_cols = [
        "policy_name",
        "scenario",
        "design_name",
        "dataset_source",
    ]

    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    for col in categorical_cols:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].astype(str).fillna("unknown")

    eps = 1e-6

    out["utilization_squared"] = out["max_utilization_ratio"] ** 2
    out["blocked_x_utilization"] = out["total_blocked_arrivals"] * out["max_utilization_ratio"]
    out["unsafe_x_utilization"] = out["total_unsafe_excess"] * out["max_utilization_ratio"]
    out["overflow_x_utilization"] = out["total_overflow_excess"] * out["max_utilization_ratio"]

    out["blocked_to_unsafe_ratio"] = out["total_blocked_arrivals"] / (
        out["total_unsafe_excess"].abs() + eps
    )
    out["overflow_to_unsafe_ratio"] = out["total_overflow_excess"] / (
        out["total_unsafe_excess"].abs() + eps
    )

    out["unsafe_momentum_1"] = out["total_unsafe_excess"] - out["total_unsafe_excess_lag1"]
    out["unsafe_momentum_2"] = out["total_unsafe_excess_lag1"] - out["total_unsafe_excess_lag2"]
    out["blocked_momentum_1"] = out["total_blocked_arrivals"] - out["total_blocked_arrivals_lag1"]
    out["blocked_momentum_2"] = out["total_blocked_arrivals_lag1"] - out["total_blocked_arrivals_lag2"]
    out["utilization_momentum_1"] = out["max_utilization_ratio"] - out["max_utilization_ratio_lag1"]
    out["overflow_momentum_1"] = out["total_overflow_excess"] - out["total_overflow_excess_lag1"]
    out["surge_gap_momentum_1"] = out["total_surge_gap"] - out["total_surge_gap_lag1"]

    out["unsafe_pressure_index"] = (
        out["total_unsafe_excess"]
        + out["total_overflow_excess"]
        + out["num_unsafe_rows"]
        + 5.0 * np.maximum(out["max_utilization_ratio"] - 1.0, 0.0)
    )

    out["access_pressure_index"] = (
        out["total_blocked_arrivals"]
        + out["total_surge_gap"]
        + 2.0 * out["blocked_high_flag"]
    )

    out["combined_stress_index"] = (
        out["unsafe_pressure_index"]
        + 0.5 * out["access_pressure_index"]
    )

    out["stress_acceleration_index"] = (
        out["unsafe_momentum_1"]
        + out["overflow_momentum_1"]
        + 2.0 * out["utilization_momentum_1"]
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def latest_state_row(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    work = df.copy()

    if "scenario" in work.columns:
        scenario_matches = work[work["scenario"].astype(str) == scenario].copy()
        if not scenario_matches.empty:
            work = scenario_matches

    if "time_index" in work.columns:
        work = work.sort_values("time_index")

    if work.empty:
        raise ValueError("No state rows available for pipeline.")

    return work.tail(1).copy()


def expected_model_columns(model: Any) -> list[str]:
    try:
        pre = model.named_steps["preprocess"]
        cols: list[str] = []
        for _, _, feature_cols in pre.transformers:
            if isinstance(feature_cols, list):
                cols.extend(feature_cols)
        return cols
    except Exception:
        return []


def prepare_X_for_model(model: Any, state_df: pd.DataFrame) -> pd.DataFrame:
    # Case 1: sklearn pipeline with ColumnTransformer
    cols = expected_model_columns(model)

    # Case 2: raw sklearn model trained directly on DataFrame
    if not cols and hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)

    if not cols:
        return state_df.copy()

    work = state_df.copy()

    for col in cols:
        if col not in work.columns:
            work[col] = 0.0

    return work[cols].copy()


def model_predict_value(model: Any, X: pd.DataFrame) -> Any:
    pred = model.predict(X)
    if isinstance(pred, (list, tuple, np.ndarray, pd.Series)):
        return pred[0]
    return pred


def model_predict_positive_probability(model: Any, X: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    probs = model.predict_proba(X)

    try:
        estimator = model.named_steps["model"]
        classes = list(estimator.classes_)
    except Exception:
        try:
            classes = list(model.classes_)
        except Exception:
            classes = []

    if 1 in classes:
        idx = classes.index(1)
    elif "1" in classes:
        idx = classes.index("1")
    else:
        idx = 1 if probs.shape[1] > 1 else 0

    return float(probs[0, idx])


def select_best_forecast_model(
    registry_df: pd.DataFrame,
    results_df: pd.DataFrame,
    task_contains: str,
    metric: str,
) -> dict[str, Any] | None:
    if registry_df.empty or results_df.empty:
        return None

    sub = results_df[
        results_df["task_name"].astype(str).str.contains(task_contains, case=False, na=False)
    ].copy()

    if sub.empty or metric not in sub.columns:
        return None

    sub = sub[sub[metric].notna()].copy()
    if sub.empty:
        return None

    best = sub.sort_values(metric, ascending=False).iloc[0]

    match = registry_df[
        (registry_df["task_name"] == best["task_name"])
        & (registry_df["model_name"] == best["model_name"])
    ].copy()

    if match.empty:
        return None

    return {
        "task_name": str(best["task_name"]),
        "model_name": str(best["model_name"]),
        "model_path": Path(str(match.iloc[0]["model_path"])),
        "metric": metric,
        "metric_value": safe_float(best.get(metric)),
    }


def select_best_regime_model(regime_df: pd.DataFrame) -> dict[str, Any] | None:
    if regime_df.empty:
        return None

    best = regime_df.sort_values("f1_macro", ascending=False).iloc[0]

    return {
        "task_name": "forecast_next_regime_strict_balanced",
        "model_name": str(best["model_name"]),
        "model_path": Path(str(best["model_path"])),
        "metric": "f1_macro",
        "metric_value": safe_float(best.get("f1_macro")),
    }


def select_best_unsafe_risk_model(risk_df: pd.DataFrame) -> dict[str, Any] | None:
    if risk_df.empty:
        return None

    if "target_col" not in risk_df.columns:
        return None

    preferred_targets = [
        "target_next_unsafe_gt_5",
        "target_next_unsafe_gt_10",
        "target_next_unsafe_top_quartile",
    ]

    chosen = None
    for target in preferred_targets:
        sub = risk_df[risk_df["target_col"].astype(str) == target].copy()
        if not sub.empty:
            chosen = sub
            break

    if chosen is None:
        chosen = risk_df.copy()

    best = chosen.sort_values(["f1_macro", "roc_auc"], ascending=False).iloc[0]

    return {
        "task_name": "unsafe_excess_risk_classifier",
        "target_col": str(best["target_col"]),
        "model_name": str(best["model_name"]),
        "model_path": Path(str(best["model_path"])),
        "metric": "f1_macro",
        "metric_value": safe_float(best.get("f1_macro")),
        "roc_auc": safe_float(best.get("roc_auc")),
        "positive_recall": safe_float(best.get("recall_pos")),
    }


def run_forecast_layer(state_df: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry = safe_read_csv(FORECAST_REGISTRY_PATH)
    forecast_results = safe_read_csv(FORECAST_RESULTS_PATH)
    regime_results = safe_read_csv(STRICT_REGIME_RESULTS_PATH)
    risk_results = safe_read_csv(UNSAFE_RISK_RESULTS_PATH)

    predictions: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []

    selected_specs = {
        "blocked_arrivals": select_best_forecast_model(
            registry, forecast_results, "blocked_arrivals", "r2"
        ),
        "unsafe_excess": select_best_forecast_model(
            registry, forecast_results, "unsafe_excess", "r2"
        ),
        "utilization_critical": select_best_forecast_model(
            registry, forecast_results, "utilization_critical", "f1_weighted"
        ),
        "regime": select_best_regime_model(regime_results),
        "unsafe_risk": select_best_unsafe_risk_model(risk_results),
    }

    for component, spec in selected_specs.items():
        if spec is None:
            predictions[component] = None
            trace.append(
                {
                    "stage": "forecast",
                    "component": component,
                    "status": "missing_model",
                }
            )
            continue

        model = load_pickle(spec["model_path"])
        X = prepare_X_for_model(model, state_df)

        if component in {"utilization_critical", "unsafe_risk"}:
            label = model_predict_value(model, X)
            prob = model_predict_positive_probability(model, X)

            predictions[component] = {
                "label": int(label) if str(label).isdigit() else str(label),
                "positive_probability": safe_float(prob),
                "model_name": spec["model_name"],
                "model_path": str(spec["model_path"]),
                "metric": spec["metric"],
                "metric_value": spec["metric_value"],
                "target_col": spec.get("target_col"),
            }

        elif component == "regime":
            label = model_predict_value(model, X)
            predictions[component] = {
                "label": str(label),
                "model_name": spec["model_name"],
                "model_path": str(spec["model_path"]),
                "metric": spec["metric"],
                "metric_value": spec["metric_value"],
            }

        else:
            value = safe_float(model_predict_value(model, X))
            predictions[component] = {
                "value": value,
                "model_name": spec["model_name"],
                "model_path": str(spec["model_path"]),
                "metric": spec["metric"],
                "metric_value": spec["metric_value"],
            }

        trace.append(
            {
                "stage": "forecast",
                "component": component,
                "status": "ok",
                "model_name": spec["model_name"],
                "model_path": str(spec["model_path"]),
                "metric": spec["metric"],
                "metric_value": spec["metric_value"],
            }
        )

    return predictions, trace


def get_policy_metrics(regime_summary: pd.DataFrame, scenario: str, policy: str) -> dict[str, float]:
    if regime_summary.empty:
        return {}

    work = regime_summary.copy()

    if "scenario" in work.columns:
        scenario_match = work[work["scenario"].astype(str) == scenario].copy()
        if not scenario_match.empty:
            work = scenario_match

    if "policy_name" in work.columns:
        work = work[work["policy_name"].astype(str) == policy].copy()

    if work.empty:
        return {}

    row = work.iloc[0]

    return {
        "total_unsafe_excess_mean": safe_float(row.get("total_unsafe_excess_mean")),
        "total_blocked_arrivals_mean": safe_float(row.get("total_blocked_arrivals_mean")),
        "max_utilization_ratio_mean": safe_float(row.get("max_utilization_ratio_mean"), 1.0),
        "num_unsafe_rows_mean": safe_float(row.get("num_unsafe_rows_mean")),
        "total_overflow_excess_mean": safe_float(row.get("total_overflow_excess_mean")),
        "total_surge_gap_mean": safe_float(row.get("total_surge_gap_mean")),
    }


def score_policy(
    policy: str,
    scenario: str,
    predictions: dict[str, Any],
    regime_summary: pd.DataFrame,
    weights: dict[str, float],
) -> dict[str, Any]:
    metrics = get_policy_metrics(regime_summary, scenario, policy)

    predicted_blocked = safe_float((predictions.get("blocked_arrivals") or {}).get("value"))
    predicted_unsafe = safe_float((predictions.get("unsafe_excess") or {}).get("value"))
    unsafe_risk_prob = safe_float((predictions.get("unsafe_risk") or {}).get("positive_probability"))
    util_critical_prob = safe_float((predictions.get("utilization_critical") or {}).get("positive_probability"))
    predicted_regime = str((predictions.get("regime") or {}).get("label", "unknown")).lower()

    base_unsafe = safe_float(metrics.get("total_unsafe_excess_mean"))
    base_blocked = safe_float(metrics.get("total_blocked_arrivals_mean"))
    base_util = safe_float(metrics.get("max_utilization_ratio_mean"), 1.0)

    crisis_indicator = 1.0 if predicted_regime == "crisis" else 0.0

    decision_score = (
        weights["unsafe_weight"] * base_unsafe
        + weights["blocked_weight"] * base_blocked
        + weights["utilization_weight"] * max(base_util - 1.0, 0.0)
        + weights["risk_weight"] * unsafe_risk_prob
        + weights["regime_crisis_weight"] * crisis_indicator
        + weights["forecast_blocked_weight"] * predicted_blocked
        + weights["forecast_unsafe_weight"] * predicted_unsafe
        + weights["forecast_utilization_risk_weight"] * util_critical_prob
    )

    return {
        "policy_name": policy,
        "scenario": scenario,
        "decision_score": float(decision_score),
        "expected_unsafe_excess": base_unsafe,
        "expected_blocked_arrivals": base_blocked,
        "expected_max_utilization": base_util,
        "expected_unsafe_rows": safe_float(metrics.get("num_unsafe_rows_mean")),
        "expected_overflow_excess": safe_float(metrics.get("total_overflow_excess_mean")),
        "expected_surge_gap": safe_float(metrics.get("total_surge_gap_mean")),
        "predicted_blocked_arrivals": predicted_blocked,
        "predicted_unsafe_excess": predicted_unsafe,
        "predicted_unsafe_risk_probability": unsafe_risk_prob,
        "predicted_utilization_critical_probability": util_critical_prob,
        "predicted_regime": predicted_regime,
    }


def select_policy(
    predictions: dict[str, Any],
    scenario: str,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if weights is None:
        weights = DEFAULT_POLICY_WEIGHTS.copy()

    regime_summary = safe_read_csv(REGIME_SUMMARY_PATH)

    rows = [
        score_policy(
            policy=policy,
            scenario=scenario,
            predictions=predictions,
            regime_summary=regime_summary,
            weights=weights,
        )
        for policy in POLICY_CANDIDATES
    ]

    score_df = pd.DataFrame(rows).sort_values("decision_score").reset_index(drop=True)

    if score_df.empty:
        raise ValueError("No policy scores could be computed.")

    return score_df, score_df.iloc[0].to_dict()


def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "high"
    if prob >= 0.40:
        return "moderate"
    return "low"


def pressure_label(x: float) -> str:
    if x >= 12:
        return "high"
    if x >= 5:
        return "moderate"
    return "low"


def build_manager_explanation(
    predictions: dict[str, Any],
    selected: dict[str, Any],
    score_df: pd.DataFrame,
) -> dict[str, str]:
    policy = str(selected["policy_name"])

    blocked = safe_float((predictions.get("blocked_arrivals") or {}).get("value"))
    unsafe = safe_float((predictions.get("unsafe_excess") or {}).get("value"))
    risk_prob = safe_float((predictions.get("unsafe_risk") or {}).get("positive_probability"))
    util_prob = safe_float((predictions.get("utilization_critical") or {}).get("positive_probability"))
    regime = str((predictions.get("regime") or {}).get("label", "unknown"))

    executive_summary = (
        f"Selected policy: {policy}. Predicted blocked arrivals = {blocked:.2f}, "
        f"predicted unsafe excess = {unsafe:.2f}, unsafe-risk probability = {risk_prob:.1%}, "
        f"utilization-critical probability = {util_prob:.1%}, predicted regime = {regime}."
    )

    if policy == "regime_robust_optimized_network":
        why_policy = (
            "The system chose regime-robust control because the combined forecast and policy-score "
            "signal favors stronger protection against overload."
        )
    elif policy == "robust_optimized_network":
        why_policy = (
            "The system chose robust network control because it offers the best safety-access balance "
            "under the current predicted risk profile."
        )
    else:
        why_policy = (
            "The system chose nominal optimized network control because access pressure and unsafe-risk "
            "signals do not justify stronger conservatism in this decision window."
        )

    if len(score_df) >= 2:
        margin = safe_float(score_df.iloc[1]["decision_score"]) - safe_float(score_df.iloc[0]["decision_score"])
    else:
        margin = 0.0

    tradeoff = (
        f"Decision-score margin versus the next-best policy = {margin:.3f}. "
        f"Unsafe-risk is {risk_label(risk_prob)}, blocked-arrival pressure is "
        f"{pressure_label(blocked)}, and utilization-critical risk is {risk_label(util_prob)}."
    )

    recommended_action = (
        "Use the selected policy for the next control window, monitor unsafe-risk probability, "
        "and escalate if blocked arrivals, unsafe risk, or utilization-critical probability increases."
    )

    return {
        "executive_summary": executive_summary,
        "why_policy": why_policy,
        "tradeoff": tradeoff,
        "recommended_action": recommended_action,
    }


def flatten_predictions(predictions: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for component, payload in predictions.items():
        if payload is None:
            rows.append({"component": component, "status": "missing"})
            continue

        row = {"component": component, "status": "ok"}
        for k, v in payload.items():
            row[k] = v
        rows.append(row)

    return pd.DataFrame(rows)


def run_pipeline(
    scenario: str = DEFAULT_SCENARIO,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    data = safe_read_csv(FORECAST_DATA_PATH)

    if data.empty:
        raise FileNotFoundError(f"Missing or empty forecast dataset: {FORECAST_DATA_PATH}")

    data = add_pipeline_engineered_features(data)
    state = latest_state_row(data, scenario=scenario)

    predictions, trace = run_forecast_layer(state)

    score_df, selected = select_policy(
        predictions=predictions,
        scenario=scenario,
        weights=weights,
    )

    manager_explanation = build_manager_explanation(
        predictions=predictions,
        selected=selected,
        score_df=score_df,
    )

    report = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario": scenario,
        "state_row_index": state.index.tolist(),
        "selected_policy": selected,
        "predictions": predictions,
        "policy_scores": score_df.to_dict(orient="records"),
        "manager_explanation": manager_explanation,
        "trace": trace,
        "notes": {
            "forecasting": "Uses leakage-safe forecasting models, strict balanced regime model, and unsafe-risk classifier.",
            "unsafe_excess": "Exact unsafe-excess regression is retained as a diagnostic but unsafe-risk classification is the decision-relevant signal.",
            "rl": "RL remains an exploratory benchmark, not the primary controller.",
        },
    }

    return report


def main() -> None:
    print("\nFULL AI-OR SYSTEM PIPELINE")
    print("=" * 70)

    report = run_pipeline(scenario=DEFAULT_SCENARIO)

    OUTPUT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    score_df = pd.DataFrame(report["policy_scores"])
    score_df.to_csv(OUTPUT_SCORE_PATH, index=False)

    trace_df = pd.DataFrame(report["trace"])
    trace_df.to_csv(OUTPUT_TRACE_PATH, index=False)

    prediction_df = flatten_predictions(report["predictions"])
    prediction_df.to_csv(OUTPUT_PREDICTION_PATH, index=False)

    print("\n=== Selected Policy ===")
    print(json.dumps(report["selected_policy"], indent=2))

    print("\n=== Predictions ===")
    print(json.dumps(report["predictions"], indent=2))

    print("\n=== Manager Explanation ===")
    print(json.dumps(report["manager_explanation"], indent=2))

    print("\n=== Policy Scores ===")
    print(score_df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_JSON_PATH)
    print(OUTPUT_SCORE_PATH)
    print(OUTPUT_TRACE_PATH)
    print(OUTPUT_PREDICTION_PATH)


if __name__ == "__main__":
    main()