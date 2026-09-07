from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def first_row_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return {}
        return obj.iloc[0].to_dict()
    return {}


# ============================================================
# Risk classification
# ============================================================

def classify_unsafe_excess(value: float) -> str:
    if value >= 10:
        return "critical"
    if value >= 5:
        return "high"
    if value > 0:
        return "moderate"
    return "low"


def classify_blocked_arrivals(value: float) -> str:
    if value >= 15:
        return "critical"
    if value >= 8:
        return "high"
    if value > 0:
        return "moderate"
    return "low"


def classify_utilization(value: float) -> str:
    if value >= 1.30:
        return "critical"
    if value >= 1.10:
        return "high"
    if value >= 1.00:
        return "elevated"
    return "stable"


def classify_regime_probabilities(
    regime_label: str,
    regime_proba: list[float] | None,
    regime_classes: list[str] | None,
) -> dict[str, Any]:
    label = safe_str(regime_label, "unknown").lower()

    if not regime_proba or not regime_classes:
        return {
            "predicted_regime": label,
            "predicted_regime_probability": None,
            "regime_confidence_label": "unknown",
        }

    mapping = {}
    for cls, prob in zip(regime_classes, regime_proba):
        mapping[safe_str(cls).lower()] = safe_float(prob)

    best_prob = mapping.get(label, max(mapping.values()) if mapping else None)

    if best_prob is None:
        conf = "unknown"
    elif best_prob >= 0.85:
        conf = "very_high"
    elif best_prob >= 0.65:
        conf = "high"
    elif best_prob >= 0.50:
        conf = "moderate"
    else:
        conf = "low"

    return {
        "predicted_regime": label,
        "predicted_regime_probability": best_prob,
        "regime_confidence_label": conf,
        "regime_probability_map": mapping,
    }


# ============================================================
# Policy reasoning
# ============================================================

def policy_reason_text(policy_name: str) -> str:
    p = safe_str(policy_name).strip().lower()

    if p == "regime_robust_optimized_network":
        return (
            "This policy emphasizes system protection under evolving stress conditions. "
            "It is designed to hedge against regime escalation and preserve safety under uncertainty."
        )
    if p == "robust_optimized_network":
        return (
            "This policy emphasizes protection against demand uncertainty using conservative allocation decisions. "
            "It prioritizes safety over access when the system is stressed."
        )
    if p == "optimized_network":
        return (
            "This policy prioritizes efficient network allocation under nominal conditions. "
            "It typically preserves more access but may be less conservative under severe uncertainty."
        )
    if p == "myopic_milp":
        return (
            "This policy optimizes short-horizon performance and reacts to immediate conditions, "
            "but may not protect as strongly against future escalation."
        )
    if p == "local_only":
        return (
            "This policy keeps decisions local and avoids broader network coordination. "
            "It may preserve autonomy but can struggle when bottlenecks become systemic."
        )
    if p == "no_transfer":
        return (
            "This policy avoids inter-hospital transfers and relies on local absorption. "
            "It can be appropriate when transfer capacity is limited but is vulnerable to local overload."
        )
    if p == "no_control":
        return (
            "This policy applies no corrective intervention. "
            "It is useful only as a benchmark and is generally not recommended during stress."
        )

    return "This policy was selected based on forecasted system conditions and optimization outputs."


def policy_tradeoff_text(
    selected_policy: str,
    predicted_unsafe_excess: float,
    predicted_blocked_arrivals: float,
) -> str:
    unsafe_label = classify_unsafe_excess(predicted_unsafe_excess)
    blocked_label = classify_blocked_arrivals(predicted_blocked_arrivals)

    if "robust" in safe_str(selected_policy).lower():
        return (
            f"The selected policy is conservative. Predicted unsafe overload remains {unsafe_label}, "
            f"but blocked arrivals are also {blocked_label}, indicating a safety-versus-access tradeoff."
        )

    return (
        f"The selected policy preserves more access when possible, but predicted unsafe overload is {unsafe_label} "
        f"and blocked arrivals are {blocked_label}, so close monitoring is still required."
    )


def recommended_action_text(
    selected_policy: str,
    predicted_regime: str,
    predicted_unsafe_excess: float,
    predicted_blocked_arrivals: float,
) -> str:
    p = safe_str(selected_policy).lower()
    regime = safe_str(predicted_regime).lower()

    if regime == "crisis" and "regime_robust" in p:
        return (
            "Activate surge-aware coordination immediately, prioritize high-risk transfers, "
            "and prepare for capacity protection over throughput."
        )
    if regime == "crisis":
        return (
            "Escalate to crisis management mode, review whether a more conservative network policy should be activated, "
            "and prepare overflow contingencies."
        )
    if predicted_unsafe_excess >= 5 and "robust" in p:
        return (
            "Maintain conservative admissions control, monitor bottleneck hospitals closely, "
            "and keep transfer capacity available for rapid reallocation."
        )
    if predicted_blocked_arrivals >= 10:
        return (
            "Monitor access degradation closely and assess whether additional ICU or transfer capacity can be activated "
            "to reduce blocked arrivals."
        )

    return (
        "Continue active monitoring, preserve flexibility in transfers and surge beds, "
        "and reassess if forecasted overload worsens."
    )


# ============================================================
# Comparison / counterfactual helpers
# ============================================================

def extract_policy_delta(
    comparison_df: pd.DataFrame | None,
    selected_policy: str,
    benchmark_policy: str,
    unsafe_col: str = "total_unsafe_excess_mean",
    blocked_col: str = "total_blocked_arrivals_mean",
) -> dict[str, Any]:
    if comparison_df is None or comparison_df.empty:
        return {}

    sdf = comparison_df.copy()
    if "policy_name" not in sdf.columns:
        return {}

    a = sdf[sdf["policy_name"] == selected_policy]
    b = sdf[sdf["policy_name"] == benchmark_policy]

    if a.empty or b.empty:
        return {}

    ra = a.iloc[0]
    rb = b.iloc[0]

    return {
        "unsafe_delta_selected_minus_benchmark": (
            safe_float(ra.get(unsafe_col)) - safe_float(rb.get(unsafe_col))
        ),
        "blocked_delta_selected_minus_benchmark": (
            safe_float(ra.get(blocked_col)) - safe_float(rb.get(blocked_col))
        ),
        "benchmark_policy": benchmark_policy,
    }


def counterfactual_text(delta_info: dict[str, Any]) -> str:
    if not delta_info:
        return "No counterfactual comparison was available."

    unsafe_delta = safe_float(delta_info.get("unsafe_delta_selected_minus_benchmark"))
    blocked_delta = safe_float(delta_info.get("blocked_delta_selected_minus_benchmark"))
    benchmark = safe_str(delta_info.get("benchmark_policy"), "benchmark policy")

    unsafe_phrase = (
        f"reduces unsafe excess by {abs(unsafe_delta):.2f}"
        if unsafe_delta < 0
        else f"increases unsafe excess by {abs(unsafe_delta):.2f}"
        if unsafe_delta > 0
        else "does not materially change unsafe excess"
    )

    blocked_phrase = (
        f"reduces blocked arrivals by {abs(blocked_delta):.2f}"
        if blocked_delta < 0
        else f"increases blocked arrivals by {abs(blocked_delta):.2f}"
        if blocked_delta > 0
        else "does not materially change blocked arrivals"
    )

    return (
        f"Relative to {benchmark}, the selected policy {unsafe_phrase} and {blocked_phrase}. "
        "This quantifies the operational tradeoff between resilience and access."
    )


# ============================================================
# Bottleneck reasoning
# ============================================================

def bottleneck_text(bottleneck_summary: pd.DataFrame | dict | None) -> str:
    row = first_row_dict(bottleneck_summary)
    if not row:
        return "No bottleneck summary was available."

    hospital = safe_str(row.get("hospital_id"), "unknown hospital")
    contribution = safe_float(row.get("contribution_share"), np.nan)
    severity = safe_str(row.get("severity"), "unknown")
    score = safe_float(row.get("bottleneck_score_mean"), np.nan)

    parts = [f"The dominant bottleneck is {hospital}"]
    if not np.isnan(contribution):
        parts.append(f"with approximately {100 * contribution:.1f}% of bottleneck contribution")
    if severity:
        parts.append(f"and severity classified as {severity.lower()}")
    if not np.isnan(score):
        parts.append(f"(score {score:.2f})")

    return " ".join(parts) + "."


# ============================================================
# Main explanation engine
# ============================================================

def explain_policy_decision(
    *,
    selected_policy: str,
    forecast: dict[str, Any],
    comparison_df: pd.DataFrame | None = None,
    bottleneck_summary: pd.DataFrame | dict | None = None,
    benchmark_policy: str = "optimized_network",
) -> dict[str, Any]:
    predicted_unsafe_excess = safe_float(forecast.get("predicted_next_unsafe_excess"))
    predicted_blocked_arrivals = safe_float(forecast.get("predicted_next_blocked_arrivals"))
    predicted_util_flag = safe_str(forecast.get("predicted_next_utilization_critical_flag"))
    predicted_regime = safe_str(forecast.get("predicted_next_regime_label"), "unknown")

    regime_info = classify_regime_probabilities(
        regime_label=predicted_regime,
        regime_proba=forecast.get("predicted_next_regime_proba"),
        regime_classes=forecast.get("predicted_next_regime_classes"),
    )

    unsafe_label = classify_unsafe_excess(predicted_unsafe_excess)
    blocked_label = classify_blocked_arrivals(predicted_blocked_arrivals)

    util_prob = None
    util_classes = forecast.get("predicted_next_utilization_classes")
    util_probs = forecast.get("predicted_next_utilization_critical_proba")
    if util_classes is not None and util_probs is not None:
        try:
            class_map = {
                safe_str(c): safe_float(p)
                for c, p in zip(util_classes, util_probs)
            }
            util_prob = class_map.get("1", None)
        except Exception:
            util_prob = None

    delta_info = extract_policy_delta(
        comparison_df=comparison_df,
        selected_policy=selected_policy,
        benchmark_policy=benchmark_policy,
    )

    executive_summary = (
        f"Forecasted next-step unsafe excess is {predicted_unsafe_excess:.2f} ({unsafe_label} risk), "
        f"and predicted blocked arrivals are {predicted_blocked_arrivals:.2f} ({blocked_label} access pressure). "
        f"The predicted next regime is {regime_info['predicted_regime']}."
    )

    if regime_info.get("predicted_regime_probability") is not None:
        executive_summary += (
            f" Regime confidence is {regime_info['regime_confidence_label'].replace('_', ' ')} "
            f"at approximately {100 * safe_float(regime_info['predicted_regime_probability']):.1f}%."
        )

    if util_prob is not None:
        executive_summary += (
            f" Predicted critical-utilization probability is approximately {100 * util_prob:.1f}%."
        )

    explanation = {
        "selected_policy": selected_policy,
        "executive_summary": executive_summary,
        "policy_reason": policy_reason_text(selected_policy),
        "tradeoff_statement": policy_tradeoff_text(
            selected_policy=selected_policy,
            predicted_unsafe_excess=predicted_unsafe_excess,
            predicted_blocked_arrivals=predicted_blocked_arrivals,
        ),
        "bottleneck_statement": bottleneck_text(bottleneck_summary),
        "counterfactual_statement": counterfactual_text(delta_info),
        "recommended_action": recommended_action_text(
            selected_policy=selected_policy,
            predicted_regime=predicted_regime,
            predicted_unsafe_excess=predicted_unsafe_excess,
            predicted_blocked_arrivals=predicted_blocked_arrivals,
        ),
        "structured_outputs": {
            "predicted_next_unsafe_excess": predicted_unsafe_excess,
            "predicted_next_blocked_arrivals": predicted_blocked_arrivals,
            "predicted_next_utilization_critical_flag": predicted_util_flag,
            "predicted_next_regime_label": predicted_regime,
            "predicted_next_regime_probability": regime_info.get("predicted_regime_probability"),
            "predicted_next_utilization_critical_probability": util_prob,
            "unsafe_risk_label": unsafe_label,
            "blocked_arrivals_label": blocked_label,
            "regime_confidence_label": regime_info.get("regime_confidence_label"),
        },
    }

    explanation["full_manager_briefing"] = "\n".join(
        [
            explanation["executive_summary"],
            explanation["policy_reason"],
            explanation["tradeoff_statement"],
            explanation["bottleneck_statement"],
            explanation["counterfactual_statement"],
            f"Recommended action: {explanation['recommended_action']}",
        ]
    )

    return explanation


# ============================================================
# Pretty printer
# ============================================================

def pretty_print_explanation(explanation: dict[str, Any]) -> None:
    print("\nPOLICY EXPLANATION")
    print("=" * 70)
    print(f"Selected policy: {explanation.get('selected_policy')}")
    print("\nExecutive summary:")
    print(explanation.get("executive_summary", ""))
    print("\nWhy this policy:")
    print(explanation.get("policy_reason", ""))
    print("\nTradeoff:")
    print(explanation.get("tradeoff_statement", ""))
    print("\nBottleneck:")
    print(explanation.get("bottleneck_statement", ""))
    print("\nCounterfactual:")
    print(explanation.get("counterfactual_statement", ""))
    print("\nRecommended action:")
    print(explanation.get("recommended_action", ""))
    print("\nFull manager briefing:")
    print(explanation.get("full_manager_briefing", ""))


# ============================================================
# Example main
# ============================================================

def main() -> None:
    forecast = {
        "predicted_next_unsafe_excess": 4.30,
        "predicted_next_blocked_arrivals": 12.11,
        "predicted_next_utilization_critical_flag": 0,
        "predicted_next_regime_label": "crisis",
        "predicted_next_utilization_critical_proba": [0.79, 0.21],
        "predicted_next_utilization_classes": ["0", "1"],
        "predicted_next_regime_proba": [0.99, 0.01],
        "predicted_next_regime_classes": ["crisis", "surge"],
    }

    comparison_df = pd.DataFrame(
        [
            {
                "policy_name": "optimized_network",
                "total_unsafe_excess_mean": 5.65,
                "total_blocked_arrivals_mean": 2.47,
            },
            {
                "policy_name": "regime_robust_optimized_network",
                "total_unsafe_excess_mean": 4.10,
                "total_blocked_arrivals_mean": 14.50,
            },
        ]
    )

    bottleneck_summary = pd.DataFrame(
        [
            {
                "hospital_id": "H3",
                "contribution_share": 0.52,
                "severity": "High",
                "bottleneck_score_mean": 8.2,
            }
        ]
    )

    explanation = explain_policy_decision(
        selected_policy="regime_robust_optimized_network",
        forecast=forecast,
        comparison_df=comparison_df,
        bottleneck_summary=bottleneck_summary,
        benchmark_policy="optimized_network",
    )

    pretty_print_explanation(explanation)


if __name__ == "__main__":
    main()