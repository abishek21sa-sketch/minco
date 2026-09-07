from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
AI_SUMMARY_DIR = RESULTS_DIR / "ai_layer_summary"
AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"
RL_DIR = RESULTS_DIR / "rl"
MANUSCRIPT_DIR = RESULTS_DIR / "manuscript_package_v2"

MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


REGIME_SUMMARY_PATH = TABLES_DIR / "regime_suite_summary.csv"
REGIME_HEADLINE_PATH = TABLES_DIR / "regime_headline_summary.csv"
SENSITIVITY_TORNADO_PATH = TABLES_DIR / "regime_sensitivity_tornado_input.csv"
STAT_HEADLINE_PATH = TABLES_DIR / "regime_final_policy_scenario_headline.csv"
REGRESSION_HEADLINE_PATH = TABLES_DIR / "final_regression_v2_headline.csv"

AI_COMPONENT_PATH = AI_SUMMARY_DIR / "ai_layer_component_summary.csv"
AI_READINESS_PATH = AI_SUMMARY_DIR / "ai_layer_readiness_summary.csv"
AI_CLAIMS_PATH = AI_SUMMARY_DIR / "ai_layer_paper_claims.csv"

FORECAST_RESULTS_PATH = AI_REPORTS_DIR / "forecast_model_results_leakage_safe.csv"
UNSAFE_RISK_RESULTS_PATH = AI_REPORTS_DIR / "unsafe_excess_risk_classifier_results.csv"
CALIBRATED_ASSIGNMENT_PATH = AI_REPORTS_DIR / "calibrated_regime_forecast_results.csv"
CALIBRATED_LAGGED_PATH = AI_REPORTS_DIR / "calibrated_regime_lagged_results.csv"

RECALIBRATION_DIST_PATH = TABLES_DIR / "regime_recalibration_distribution.csv"
RECALIBRATION_RECOMMENDATION_PATH = TABLES_DIR / "regime_recalibration_recommendations.csv"

RL_CONTROLLER_PATH = RL_DIR / "q_learning_v2_controller_summary.csv"


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


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def best_by_metric(df: pd.DataFrame, metric: str) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    work = df[df[metric].notna()].copy()
    if work.empty:
        return None
    return work.sort_values(metric, ascending=False).iloc[0]


def build_headline_findings(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    regime_headline = inputs["regime_headline"]
    if not regime_headline.empty and "regime_vs_nominal_gain" in regime_headline.columns:
        wins = int((regime_headline["regime_vs_nominal_gain"] > 0).sum())
        total = int(len(regime_headline))
        avg_gain = safe_float(regime_headline["regime_vs_nominal_gain"].mean())
        rows.append({
            "finding_id": "F1",
            "theme": "regime_robust_policy",
            "headline": f"Regime-robust optimization improved unsafe excess versus nominal optimization in {wins}/{total} scenarios.",
            "evidence": f"Average unsafe-excess gain versus nominal = {avg_gain:.3f}.",
            "paper_section": "Results",
        })

    recal = inputs["recalibration_dist"]
    if not recal.empty:
        original = recal[recal["label_source"] == "original_next_regime_label"]
        score = recal[recal["label_source"] == "calibrated_regime_label_score"]

        def crisis_pct(df: pd.DataFrame) -> float:
            row = df[df["regime"].astype(str).str.lower() == "crisis"]
            return safe_float(row.iloc[0]["percentage"]) if not row.empty else np.nan

        rows.append({
            "finding_id": "F1b",
            "theme": "regime_calibration",
            "headline": "Operational recalibration reduced crisis dominance in the regime labels.",
            "evidence": (
                f"Original crisis share = {crisis_pct(original):.3f}; "
                f"score-calibrated crisis share = {crisis_pct(score):.3f}."
            ),
            "paper_section": "Regime calibration",
        })

    sens = inputs["sensitivity_tornado"]
    if not sens.empty and "unsafe_excess_change_vs_base" in sens.columns:
        work = sens[sens.get("design_name", "") != "base"].copy() if "design_name" in sens.columns else sens.copy()
        if not work.empty:
            work["abs_change"] = work["unsafe_excess_change_vs_base"].abs()
            row = work.sort_values("abs_change", ascending=False).iloc[0]
            rows.append({
                "finding_id": "F2",
                "theme": "sensitivity",
                "headline": f"The strongest sensitivity lever was {row.get('design_name', 'largest_sensitivity_lever')}.",
                "evidence": (
                    f"Unsafe-excess change versus base = {safe_float(row.get('unsafe_excess_change_vs_base')):.3f}; "
                    f"blocked-arrivals change versus base = {safe_float(row.get('blocked_arrivals_change_vs_base')):.3f}."
                ),
                "paper_section": "Sensitivity analysis",
            })

    forecast = inputs["forecast_results"]
    if not forecast.empty:
        blocked = forecast[forecast["task_name"].astype(str).str.contains("blocked_arrivals", case=False, na=False)]
        best = best_by_metric(blocked, "r2")
        if best is not None:
            rows.append({
                "finding_id": "F3",
                "theme": "forecast_ai",
                "headline": "The leakage-controlled forecasting layer predicted next-period blocked arrivals accurately.",
                "evidence": f"Best model = {best['model_name']}; R2 = {safe_float(best.get('r2')):.3f}; MAE = {safe_float(best.get('mae')):.3f}.",
                "paper_section": "AI forecasting layer",
            })

        unsafe = forecast[forecast["task_name"].astype(str).str.contains("unsafe_excess", case=False, na=False)]
        best_u = best_by_metric(unsafe, "r2")
        if best_u is not None:
            rows.append({
                "finding_id": "F3b",
                "theme": "forecast_ai",
                "headline": "Exact unsafe-excess magnitude forecasting remained difficult.",
                "evidence": f"Best leakage-safe unsafe-excess model = {best_u['model_name']}; R2 = {safe_float(best_u.get('r2')):.3f}; MAE = {safe_float(best_u.get('mae')):.3f}.",
                "paper_section": "AI forecasting layer",
            })

    risk = inputs["unsafe_risk"]
    if not risk.empty:
        best_targets = []
        for target in sorted(risk["target_col"].dropna().unique()):
            sub = risk[risk["target_col"] == target].copy()
            best_targets.append(sub.sort_values(["f1_macro", "roc_auc"], ascending=False).iloc[0])
        bt = pd.DataFrame(best_targets)
        best = bt.sort_values(["f1_macro", "roc_auc"], ascending=False).iloc[0]
        rows.append({
            "finding_id": "F3c",
            "theme": "forecast_ai",
            "headline": "Unsafe-excess threshold risk was more predictable than exact unsafe-excess magnitude.",
            "evidence": (
                f"Best risk target = {best['target_col']}; model = {best['model_name']}; "
                f"macro-F1 = {safe_float(best.get('f1_macro')):.3f}; "
                f"positive recall = {safe_float(best.get('recall_pos')):.3f}; "
                f"AUC = {safe_float(best.get('roc_auc')):.3f}."
            ),
            "paper_section": "AI forecasting layer",
        })

    assign = inputs["calibrated_assignment"]
    best_a = best_by_metric(assign, "f1_macro")
    if best_a is not None:
        rows.append({
            "finding_id": "F3d",
            "theme": "regime_ai",
            "headline": "Calibrated regimes can be assigned reliably from current operational state.",
            "evidence": (
                f"Best current-state assignment model = {best_a['model_name']}; "
                f"macro-F1 = {safe_float(best_a.get('f1_macro')):.3f}. "
                f"This is treated as state assignment, not true forecasting."
            ),
            "paper_section": "AI forecasting layer",
        })

    lag = inputs["calibrated_lagged"]
    best_l = best_by_metric(lag, "f1_macro")
    if best_l is not None:
        rows.append({
            "finding_id": "F3e",
            "theme": "regime_ai",
            "headline": "Strict lagged calibrated-regime forecasting remained limited.",
            "evidence": (
                f"Best lagged-only model = {best_l['model_name']}; "
                f"macro-F1 = {safe_float(best_l.get('f1_macro')):.3f}; "
                f"weighted-F1 = {safe_float(best_l.get('f1_weighted')):.3f}."
            ),
            "paper_section": "AI forecasting layer",
        })

    ai = inputs["ai_component"]
    if not ai.empty:
        xai = ai[(ai["ai_layer"] == "explainable_ai") & (ai["paper_readiness"] == "paper_ready")]
        rows.append({
            "finding_id": "F4",
            "theme": "explainable_ai",
            "headline": f"The explainable AI layer has {len(xai)} paper-ready components.",
            "evidence": "Includes manager explanations, policy counterfactuals, and sensitivity counterfactuals.",
            "paper_section": "Explainable decision support",
        })

    rl = inputs["rl_controller"]
    if not rl.empty and "mean_reward" in rl.columns:
        ranked = rl.sort_values("mean_reward", ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        learned = ranked[ranked["policy_name"] == "learned_q_policy"]
        if not learned.empty:
            rank = int(learned.index[0] + 1)
            rows.append({
                "finding_id": "F5",
                "theme": "rl_benchmark",
                "headline": f"The RL meta-controller ranked {rank}/{len(ranked)} in the current benchmark.",
                "evidence": (
                    f"Best controller = {best['policy_name']} with mean reward {safe_float(best.get('mean_reward')):.3f}; "
                    f"RL mean reward = {safe_float(learned.iloc[0].get('mean_reward')):.3f}."
                ),
                "paper_section": "RL benchmark",
            })

    return pd.DataFrame(rows)


def copy_selected_tables(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table_specs = [
        ("T1_regime_policy_summary.csv", "Main regime-policy summary", "regime_summary"),
        ("T2_regime_headline_summary.csv", "Regime robust versus nominal headline results", "regime_headline"),
        ("T3_sensitivity_tornado_input.csv", "Sensitivity-analysis tornado inputs", "sensitivity_tornado"),
        ("T4_policy_significance_headline.csv", "Policy significance headline table", "stat_headline"),
        ("T5_regression_headline.csv", "Regression headline table", "regression_headline"),
        ("T6_ai_layer_component_summary.csv", "AI layer component readiness", "ai_component"),
        ("T7_rl_controller_summary.csv", "RL versus optimizer controller benchmark", "rl_controller"),
        ("T8_regime_recalibration_distribution.csv", "Original versus calibrated regime distribution", "recalibration_dist"),
        ("T9_calibrated_regime_lagged_results.csv", "Strict lagged calibrated-regime forecast results", "calibrated_lagged"),
    ]

    rows = []
    for filename, description, key in table_specs:
        df = inputs.get(key, pd.DataFrame())
        out_path = MANUSCRIPT_DIR / filename
        status = "missing_or_empty"
        if not df.empty:
            df.to_csv(out_path, index=False)
            status = "created"
        rows.append({
            "table_file": filename,
            "description": description,
            "source_key": key,
            "status": status,
            "n_rows": len(df),
            "n_cols": len(df.columns) if not df.empty else 0,
        })
    return pd.DataFrame(rows)


def build_figure_checklist() -> pd.DataFrame:
    expected = [
        ("regime_publication_tradeoff_frontier.png", "Safety-access tradeoff frontier", "Main results"),
        ("regime_publication_unsafe_heatmap.png", "Unsafe excess heatmap", "Main results"),
        ("regime_publication_regime_advantage_bars.png", "Regime robust advantage bars", "Main results"),
        ("regime_sensitivity_tornado_unsafe.png", "Unsafe-excess sensitivity tornado chart", "Sensitivity"),
        ("regime_sensitivity_tornado_blocked.png", "Blocked-arrivals sensitivity tornado chart", "Sensitivity"),
        ("regime_sensitivity_tradeoff_scatter.png", "Sensitivity safety-access scatter", "Sensitivity"),
        ("regime_sensitivity_lever_importance.png", "Sensitivity lever importance bars", "Sensitivity"),
        ("unsafe_excess_risk_target_distribution.png", "Unsafe-risk target distribution", "AI forecasting"),
        ("unsafe_excess_risk_feature_importance.png", "Unsafe-risk feature importance", "AI forecasting"),
        ("regime_recalibration_distribution.png", "Original versus calibrated regime distribution", "Regime calibration"),
        ("regime_recalibration_by_scenario.png", "Calibrated regime distribution by scenario", "Regime calibration"),
    ]
    return pd.DataFrame([
        {
            "figure_file": f,
            "description": d,
            "paper_section": s,
            "exists": (FIGURES_DIR / f).exists(),
            "path": str(FIGURES_DIR / f),
        }
        for f, d, s in expected
    ])


def build_manuscript_outline() -> str:
    return """# Manuscript Package v2

## Working Title

AI-Augmented Regime-Calibrated Robust Optimization for Hospital Network Capacity Control Under Demand Uncertainty

## Central Thesis

A regime-calibrated robust optimization framework can support hospital network transfer and capacity decisions under stochastic demand by combining operational regime assignment, robust MILP control, simulation validation, unsafe-risk classification, and explainable counterfactual decision support.

## Important Framing Correction

The original regime labels were crisis-dominant. The manuscript should therefore use calibrated operational regime labels for regime analysis and clearly distinguish between:
1. current-state calibrated regime assignment, and
2. strict lagged regime forecasting.

The current-state calibrated regime assignment is strong because it identifies the regime from operational state variables. The strict lagged regime forecast is limited, so it should not be overstated as a major forecasting contribution.

## Recommended Contributions

1. Regime-calibrated hospital network optimization framework.
2. Robust and regime-aware MILP control for safety-access tradeoffs.
3. Stochastic simulation testbed with sensitivity and policy comparison.
4. Leakage-controlled forecasting layer for blocked arrivals and unsafe-risk threshold alerts.
5. Explainable counterfactual decision-support layer.
6. RL benchmark showing structured robust optimization remains stronger than shallow tabular RL in this setting.

## Results Sections

1. Policy comparison and safety-access tradeoff.
2. Regime label audit and recalibration.
3. Sensitivity analysis.
4. Leakage-controlled forecasting results.
5. Unsafe-risk threshold classification.
6. Calibrated regime assignment versus strict lagged regime forecast.
7. Explainable AI and counterfactual decision support.
8. RL benchmark.
"""


def build_limitations_text() -> str:
    return """# Limitations and Safe Claims

## Safe Claims

- The original regime labels were crisis-dominant and were audited before manuscript framing.
- Calibrated operational regime labels reduce crisis dominance and provide a more defensible regime definition.
- Current-state calibrated regime assignment is strong, but this should be interpreted as state classification, not true future forecasting.
- Strict lagged calibrated-regime forecasting is limited, indicating regime transition prediction remains challenging.
- Blocked-arrival forecasting is strong.
- Exact unsafe-excess regression is weak, while unsafe-risk threshold classification provides moderate signal.
- RL is an exploratory benchmark and does not dominate the best robust optimization baseline.

## Claims to Avoid

- Do not claim broad regime-switching validation from the original labels alone.
- Do not claim F1=1.000 calibrated regime assignment as true forecasting.
- Do not claim exact unsafe-excess regression is strong.
- Do not claim RL is the best-performing method.
- Do not claim clinical deployment readiness without real hospital validation.

## Recommended Manuscript Framing

The headline method should be a regime-calibrated, forecast-augmented robust optimization framework with explainable counterfactual decision support. The AI layer should be framed as operational support: blocked-arrival prediction, unsafe-risk alerts, calibrated regime assignment, and decision explanation.
"""


def build_abstract_evidence_bullets(headlines: pd.DataFrame) -> str:
    lines = ["# Abstract Evidence Bullets", ""]
    if headlines.empty:
        lines.append("No headline findings available.")
    else:
        for _, row in headlines.iterrows():
            lines.append(f"- {row['headline']} {row['evidence']}")
    return "\n".join(lines)


def main() -> None:
    inputs = {
        "regime_summary": safe_read_csv(REGIME_SUMMARY_PATH),
        "regime_headline": safe_read_csv(REGIME_HEADLINE_PATH),
        "sensitivity_tornado": safe_read_csv(SENSITIVITY_TORNADO_PATH),
        "stat_headline": safe_read_csv(STAT_HEADLINE_PATH),
        "regression_headline": safe_read_csv(REGRESSION_HEADLINE_PATH),
        "ai_component": safe_read_csv(AI_COMPONENT_PATH),
        "ai_readiness": safe_read_csv(AI_READINESS_PATH),
        "ai_claims": safe_read_csv(AI_CLAIMS_PATH),
        "forecast_results": safe_read_csv(FORECAST_RESULTS_PATH),
        "unsafe_risk": safe_read_csv(UNSAFE_RISK_RESULTS_PATH),
        "calibrated_assignment": safe_read_csv(CALIBRATED_ASSIGNMENT_PATH),
        "calibrated_lagged": safe_read_csv(CALIBRATED_LAGGED_PATH),
        "recalibration_dist": safe_read_csv(RECALIBRATION_DIST_PATH),
        "recalibration_recommendations": safe_read_csv(RECALIBRATION_RECOMMENDATION_PATH),
        "rl_controller": safe_read_csv(RL_CONTROLLER_PATH),
    }

    headlines = build_headline_findings(inputs)
    table_manifest = copy_selected_tables(inputs)
    figure_checklist = build_figure_checklist()

    headline_path = MANUSCRIPT_DIR / "headline_findings.csv"
    table_manifest_path = MANUSCRIPT_DIR / "table_manifest.csv"
    figure_checklist_path = MANUSCRIPT_DIR / "figure_checklist.csv"
    outline_path = MANUSCRIPT_DIR / "manuscript_outline.md"
    limitations_path = MANUSCRIPT_DIR / "limitations_and_safe_claims.md"
    abstract_path = MANUSCRIPT_DIR / "abstract_evidence_bullets.md"
    package_summary_path = MANUSCRIPT_DIR / "package_summary.json"

    headlines.to_csv(headline_path, index=False)
    table_manifest.to_csv(table_manifest_path, index=False)
    figure_checklist.to_csv(figure_checklist_path, index=False)

    write_text(outline_path, build_manuscript_outline())
    write_text(limitations_path, build_limitations_text())
    write_text(abstract_path, build_abstract_evidence_bullets(headlines))

    summary = {
        "n_headline_findings": int(len(headlines)),
        "n_tables_created": int((table_manifest["status"] == "created").sum()),
        "n_tables_missing_or_empty": int((table_manifest["status"] != "created").sum()),
        "n_expected_figures": int(len(figure_checklist)),
        "n_figures_existing": int(figure_checklist["exists"].sum()),
        "output_dir": str(MANUSCRIPT_DIR),
    }

    package_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nMANUSCRIPT PACKAGE V2")
    print("=" * 70)
    print("\n=== Headline Findings ===")
    print(headlines.to_string(index=False))
    print("\n=== Table Manifest ===")
    print(table_manifest.to_string(index=False))
    print("\n=== Figure Checklist ===")
    print(figure_checklist.to_string(index=False))
    print("\nSaved:")
    print(headline_path)
    print(table_manifest_path)
    print(figure_checklist_path)
    print(outline_path)
    print(limitations_path)
    print(abstract_path)
    print(package_summary_path)


if __name__ == "__main__":
    main()