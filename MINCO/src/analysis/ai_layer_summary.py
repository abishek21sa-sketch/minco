from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"
RL_DIR = RESULTS_DIR / "rl"

SUMMARY_OUT_DIR = RESULTS_DIR / "ai_layer_summary"
SUMMARY_OUT_DIR.mkdir(parents=True, exist_ok=True)


FORECAST_RESULTS_PATH = AI_REPORTS_DIR / "forecast_model_results_leakage_safe.csv"
STRICT_REGIME_RESULTS_PATH = AI_REPORTS_DIR / "strict_balanced_regime_results.csv"
UNSAFE_RISK_RESULTS_PATH = AI_REPORTS_DIR / "unsafe_excess_risk_classifier_results.csv"

CALIBRATED_REGIME_ASSIGNMENT_PATH = AI_REPORTS_DIR / "calibrated_regime_forecast_results.csv"
CALIBRATED_REGIME_LAGGED_PATH = AI_REPORTS_DIR / "calibrated_regime_lagged_results.csv"

COUNTERFACTUAL_POLICY_PATH = TABLES_DIR / "counterfactual_policy_comparisons.csv"
COUNTERFACTUAL_SENSITIVITY_PATH = TABLES_DIR / "counterfactual_sensitivity_comparisons.csv"

RL_CONTROLLER_SUMMARY_PATH = RL_DIR / "q_learning_v2_controller_summary.csv"
RL_HEADLINE_PATH = RL_DIR / "q_learning_v2_headline.csv"
RL_HISTORY_PATH = RL_DIR / "q_learning_v2_training_history.csv"


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def readiness_from_score(score: float, paper_threshold=0.70, usable_threshold=0.55) -> str:
    if score >= paper_threshold:
        return "paper_ready"
    if score >= usable_threshold:
        return "usable_but_needs_strengthening"
    return "prototype_or_supporting_only"


def best_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    work = df[df[metric].notna()].copy()
    if work.empty:
        return None
    return work.sort_values(metric, ascending=False).iloc[0]


def build_forecast_summary(
    forecast_df: pd.DataFrame,
    strict_regime_df: pd.DataFrame,
    unsafe_risk_df: pd.DataFrame,
    calibrated_assignment_df: pd.DataFrame,
    calibrated_lagged_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if forecast_df.empty:
        rows.append({
            "ai_layer": "forecast_ai",
            "component": "leakage_safe_forecast_models",
            "status": "missing",
            "evidence": "forecast_model_results_leakage_safe.csv not found",
            "paper_readiness": "not_ready",
        })
    else:
        for task in sorted(forecast_df["task_name"].dropna().unique()):
            if "regime" in task:
                continue

            sub = forecast_df[forecast_df["task_name"] == task].copy()

            if "r2" in sub.columns and sub["r2"].notna().any():
                best = best_row(sub, "r2")
                r2 = safe_float(best.get("r2"))
                readiness = "paper_ready" if r2 >= 0.70 else (
                    "usable_but_needs_strengthening" if r2 >= 0.30 else "prototype_or_supporting_only"
                )
                evidence = (
                    f"best_model={best['model_name']}; "
                    f"r2={r2:.3f}; mae={safe_float(best.get('mae')):.3f}; "
                    f"rmse={safe_float(best.get('rmse')):.3f}; leakage_safe=True"
                )
            elif "f1_weighted" in sub.columns and sub["f1_weighted"].notna().any():
                best = best_row(sub, "f1_weighted")
                f1w = safe_float(best.get("f1_weighted"))
                readiness = "paper_ready" if f1w >= 0.80 else (
                    "usable_but_needs_strengthening" if f1w >= 0.65 else "prototype_or_supporting_only"
                )
                evidence = (
                    f"best_model={best['model_name']}; "
                    f"accuracy={safe_float(best.get('accuracy')):.3f}; "
                    f"f1_macro={safe_float(best.get('f1_macro')):.3f}; "
                    f"f1_weighted={f1w:.3f}; leakage_safe=True"
                )
            else:
                best = sub.iloc[0]
                readiness = "unknown"
                evidence = f"task={task}; model={best.get('model_name', 'unknown')}; leakage_safe=True"

            rows.append({
                "ai_layer": "forecast_ai",
                "component": task,
                "status": "implemented",
                "evidence": evidence,
                "paper_readiness": readiness,
            })

    if not unsafe_risk_df.empty:
        best_targets = []
        for target_col in sorted(unsafe_risk_df["target_col"].dropna().unique()):
            sub = unsafe_risk_df[unsafe_risk_df["target_col"] == target_col].copy()
            best = sub.sort_values(["f1_macro", "roc_auc"], ascending=False).iloc[0]
            best_targets.append(best)

        bt = pd.DataFrame(best_targets)
        strongest = bt.sort_values(["f1_macro", "roc_auc"], ascending=False).iloc[0]
        f1m = safe_float(strongest.get("f1_macro"))

        evidence = " | ".join([
            (
                f"{r['target_col']}: best={r['model_name']}, "
                f"f1_macro={safe_float(r.get('f1_macro')):.3f}, "
                f"f1_pos={safe_float(r.get('f1_pos')):.3f}, "
                f"recall_pos={safe_float(r.get('recall_pos')):.3f}, "
                f"auc={safe_float(r.get('roc_auc')):.3f}"
            )
            for _, r in bt.iterrows()
        ])

        rows.append({
            "ai_layer": "forecast_ai",
            "component": "unsafe_excess_risk_classifier",
            "status": "implemented",
            "evidence": evidence + "; leakage_safe=True",
            "paper_readiness": readiness_from_score(f1m, 0.65, 0.55),
        })
    else:
        rows.append({
            "ai_layer": "forecast_ai",
            "component": "unsafe_excess_risk_classifier",
            "status": "missing",
            "evidence": "unsafe_excess_risk_classifier_results.csv not found",
            "paper_readiness": "not_ready",
        })

    if not calibrated_assignment_df.empty:
        best = best_row(calibrated_assignment_df, "f1_macro")
        f1m = safe_float(best.get("f1_macro"))
        rows.append({
            "ai_layer": "forecast_ai",
            "component": "calibrated_regime_assignment_current_state",
            "status": "implemented",
            "evidence": (
                f"best_model={best['model_name']}; accuracy={safe_float(best.get('accuracy')):.3f}; "
                f"f1_macro={f1m:.3f}; f1_weighted={safe_float(best.get('f1_weighted')):.3f}; "
                f"interpretation=state_assignment_not_true_forecast"
            ),
            "paper_readiness": "paper_ready_as_state_assignment",
        })
    else:
        rows.append({
            "ai_layer": "forecast_ai",
            "component": "calibrated_regime_assignment_current_state",
            "status": "missing",
            "evidence": "calibrated_regime_forecast_results.csv not found",
            "paper_readiness": "not_ready",
        })

    if not calibrated_lagged_df.empty:
        best = best_row(calibrated_lagged_df, "f1_macro")
        f1m = safe_float(best.get("f1_macro"))
        rows.append({
            "ai_layer": "forecast_ai",
            "component": "calibrated_regime_forecast_strict_lagged",
            "status": "implemented",
            "evidence": (
                f"best_model={best['model_name']}; accuracy={safe_float(best.get('accuracy')):.3f}; "
                f"f1_macro={f1m:.3f}; f1_weighted={safe_float(best.get('f1_weighted')):.3f}; "
                f"recall_macro={safe_float(best.get('recall_macro')):.3f}; strict_lagged_only=True"
            ),
            "paper_readiness": readiness_from_score(f1m, 0.60, 0.40),
        })
    else:
        rows.append({
            "ai_layer": "forecast_ai",
            "component": "calibrated_regime_forecast_strict_lagged",
            "status": "missing",
            "evidence": "calibrated_regime_lagged_results.csv not found",
            "paper_readiness": "not_ready",
        })

    if not strict_regime_df.empty:
        best = best_row(strict_regime_df, "f1_macro")
        rows.append({
            "ai_layer": "forecast_ai",
            "component": "original_regime_strict_balanced_deprecated",
            "status": "implemented",
            "evidence": (
                f"best_model={best['model_name']}; accuracy={safe_float(best.get('accuracy')):.3f}; "
                f"f1_macro={safe_float(best.get('f1_macro')):.3f}; "
                f"kept_for_audit_not_main_claim=True"
            ),
            "paper_readiness": "supporting_audit_only",
        })

    return pd.DataFrame(rows)


def build_xai_summary(policy_cf_df: pd.DataFrame, sensitivity_cf_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "ai_layer": "explainable_ai",
        "component": "policy_counterfactuals",
        "status": "implemented" if not policy_cf_df.empty else "missing",
        "evidence": (
            f"{len(policy_cf_df)} metric rows across "
            f"{policy_cf_df[['selected_policy', 'benchmark_policy']].drop_duplicates().shape[0] if not policy_cf_df.empty and {'selected_policy','benchmark_policy'}.issubset(policy_cf_df.columns) else 0} policy comparisons"
            if not policy_cf_df.empty else "counterfactual_policy_comparisons.csv not found"
        ),
        "paper_readiness": "paper_ready" if not policy_cf_df.empty else "not_ready",
    })

    rows.append({
        "ai_layer": "explainable_ai",
        "component": "sensitivity_counterfactuals",
        "status": "implemented" if not sensitivity_cf_df.empty else "missing",
        "evidence": (
            f"{len(sensitivity_cf_df)} metric rows across "
            f"{sensitivity_cf_df[['selected_design', 'benchmark_design']].drop_duplicates().shape[0] if not sensitivity_cf_df.empty and {'selected_design','benchmark_design'}.issubset(sensitivity_cf_df.columns) else 0} sensitivity comparisons"
            if not sensitivity_cf_df.empty else "counterfactual_sensitivity_comparisons.csv not found"
        ),
        "paper_readiness": "paper_ready" if not sensitivity_cf_df.empty else "not_ready",
    })

    rows.append({
        "ai_layer": "explainable_ai",
        "component": "manager_explanation_generator",
        "status": "implemented",
        "evidence": "policy_explainer.py and run_full_pipeline.py generate manager-facing explanations, tradeoff statements, and recommended actions",
        "paper_readiness": "paper_ready",
    })

    return pd.DataFrame(rows)


def build_rl_summary(controller_df: pd.DataFrame, headline_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    if controller_df.empty:
        return pd.DataFrame([{
            "ai_layer": "rl_benchmark",
            "component": "q_learning_meta_controller",
            "status": "missing",
            "evidence": "q_learning_v2_controller_summary.csv not found",
            "paper_readiness": "not_ready",
        }])

    rows = []
    ranked = controller_df.sort_values("mean_reward", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    best = ranked.iloc[0]
    rl = ranked[ranked["policy_name"] == "learned_q_policy"]

    if rl.empty:
        evidence = "learned_q_policy not found"
    else:
        r = rl.iloc[0]
        evidence = (
            f"learned_q_policy_rank={int(r['rank'])}/{len(ranked)}; "
            f"rl_mean_reward={safe_float(r.get('mean_reward')):.3f}; "
            f"best_policy={best['policy_name']}; best_mean_reward={safe_float(best.get('mean_reward')):.3f}; "
            f"interpretation=exploratory_benchmark_only"
        )

    rows.append({
        "ai_layer": "rl_benchmark",
        "component": "q_learning_meta_controller",
        "status": "implemented",
        "evidence": evidence,
        "paper_readiness": "paper_ready_as_benchmark",
    })

    if not history_df.empty:
        rows.append({
            "ai_layer": "rl_benchmark",
            "component": "training_curve",
            "status": "implemented",
            "evidence": (
                f"n_episodes={len(history_df)}; "
                f"initial_reward={safe_float(history_df.iloc[0].get('episode_reward')):.3f}; "
                f"final_reward={safe_float(history_df.iloc[-1].get('episode_reward')):.3f}; "
                f"final_epsilon={safe_float(history_df.iloc[-1].get('final_epsilon')):.3f}; "
                f"interpretation=diagnostic_only"
            ),
            "paper_readiness": "paper_ready_as_benchmark",
        })

    if not headline_df.empty and "best_policy" in headline_df.columns:
        rows.append({
            "ai_layer": "rl_benchmark",
            "component": "regime_winners",
            "status": "implemented",
            "evidence": json.dumps(headline_df["best_policy"].value_counts().to_dict()),
            "paper_readiness": "paper_ready_as_benchmark",
        })

    return pd.DataFrame(rows)


def build_overall_readiness(layer_df: pd.DataFrame) -> pd.DataFrame:
    readiness_order = {
        "paper_ready": 4,
        "paper_ready_as_state_assignment": 4,
        "paper_ready_as_benchmark": 3,
        "usable_but_needs_strengthening": 2,
        "prototype_or_supporting_only": 1,
        "supporting_audit_only": 1,
        "not_ready": 0,
        "unknown": 0,
    }

    work = layer_df.copy()
    work["readiness_score"] = work["paper_readiness"].map(readiness_order).fillna(0)

    rows = []
    for layer in sorted(work["ai_layer"].dropna().unique()):
        sub = work[work["ai_layer"] == layer]
        rows.append({
            "ai_layer": layer,
            "n_components": len(sub),
            "min_readiness_score": float(sub["readiness_score"].min()),
            "mean_readiness_score": float(sub["readiness_score"].mean()),
            "max_readiness_score": float(sub["readiness_score"].max()),
            "dominant_status": sub["status"].mode().iloc[0] if not sub["status"].mode().empty else "unknown",
            "lowest_readiness_label": sub.sort_values("readiness_score").iloc[0]["paper_readiness"],
            "highest_readiness_label": sub.sort_values("readiness_score", ascending=False).iloc[0]["paper_readiness"],
        })

    return pd.DataFrame(rows)


def build_paper_claims(layer_df: pd.DataFrame) -> pd.DataFrame:
    claims = [
        {
            "claim_type": "safe_claim",
            "claim": "The project uses calibrated operational regime labels to correct crisis dominance in the original regime labels.",
            "supporting_layer": "forecast_ai",
        },
        {
            "claim_type": "safe_claim",
            "claim": "Calibrated regimes can be assigned reliably from current operational state, but this is state assignment, not a true forward forecast.",
            "supporting_layer": "forecast_ai",
        },
        {
            "claim_type": "safe_claim",
            "claim": "Strict lagged calibrated-regime forecasting is limited, indicating that regime transitions are harder to predict from history alone.",
            "supporting_layer": "forecast_ai",
        },
        {
            "claim_type": "safe_claim",
            "claim": "Blocked-arrival forecasting is the strongest supervised forecasting component.",
            "supporting_layer": "forecast_ai",
        },
        {
            "claim_type": "safe_claim",
            "claim": "Exact unsafe-excess regression is weak, but unsafe-risk threshold classification provides moderate operational warning signal.",
            "supporting_layer": "forecast_ai",
        },
        {
            "claim_type": "safe_claim",
            "claim": "The explainable AI layer converts forecasts, policy comparisons, and bottleneck information into manager-facing recommendations.",
            "supporting_layer": "explainable_ai",
        },
        {
            "claim_type": "safe_claim",
            "claim": "Reinforcement learning is evaluated as an exploratory benchmark and should not be framed as the primary controller.",
            "supporting_layer": "rl_benchmark",
        },
        {
            "claim_type": "avoid_claim",
            "claim": "Do not claim F1=1.000 calibrated regime results as true forecasting performance because current-state assignment uses variables that define the calibrated label.",
            "supporting_layer": "forecast_ai",
        },
        {
            "claim_type": "avoid_claim",
            "claim": "Do not claim RL is the best-performing method.",
            "supporting_layer": "rl_benchmark",
        },
    ]
    return pd.DataFrame(claims)


def main() -> None:
    forecast_df = safe_read_csv(FORECAST_RESULTS_PATH)
    strict_regime_df = safe_read_csv(STRICT_REGIME_RESULTS_PATH)
    unsafe_risk_df = safe_read_csv(UNSAFE_RISK_RESULTS_PATH)
    calibrated_assignment_df = safe_read_csv(CALIBRATED_REGIME_ASSIGNMENT_PATH)
    calibrated_lagged_df = safe_read_csv(CALIBRATED_REGIME_LAGGED_PATH)

    policy_cf = safe_read_csv(COUNTERFACTUAL_POLICY_PATH)
    sensitivity_cf = safe_read_csv(COUNTERFACTUAL_SENSITIVITY_PATH)

    rl_controller = safe_read_csv(RL_CONTROLLER_SUMMARY_PATH)
    rl_headline = safe_read_csv(RL_HEADLINE_PATH)
    rl_history = safe_read_csv(RL_HISTORY_PATH)

    forecast_summary = build_forecast_summary(
        forecast_df=forecast_df,
        strict_regime_df=strict_regime_df,
        unsafe_risk_df=unsafe_risk_df,
        calibrated_assignment_df=calibrated_assignment_df,
        calibrated_lagged_df=calibrated_lagged_df,
    )
    xai_summary = build_xai_summary(policy_cf, sensitivity_cf)
    rl_summary = build_rl_summary(rl_controller, rl_headline, rl_history)

    layer_summary = pd.concat([forecast_summary, xai_summary, rl_summary], ignore_index=True)
    readiness = build_overall_readiness(layer_summary)
    claims = build_paper_claims(layer_summary)

    layer_summary_path = SUMMARY_OUT_DIR / "ai_layer_component_summary.csv"
    readiness_path = SUMMARY_OUT_DIR / "ai_layer_readiness_summary.csv"
    claims_path = SUMMARY_OUT_DIR / "ai_layer_paper_claims.csv"

    layer_summary.to_csv(layer_summary_path, index=False)
    readiness.to_csv(readiness_path, index=False)
    claims.to_csv(claims_path, index=False)

    print("\nAI LAYER SUMMARY")
    print("=" * 70)
    print("\n=== Component Summary ===")
    print(layer_summary.to_string(index=False))
    print("\n=== Readiness Summary ===")
    print(readiness.to_string(index=False))
    print("\n=== Paper Claims Guidance ===")
    print(claims.to_string(index=False))
    print("\nSaved:")
    print(layer_summary_path)
    print(readiness_path)
    print(claims_path)


if __name__ == "__main__":
    main()