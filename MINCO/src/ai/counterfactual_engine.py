from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"

REGIME_SUMMARY_PATH = TABLES_DIR / "regime_suite_summary.csv"
SENSITIVITY_SUMMARY_PATH = TABLES_DIR / "regime_sensitivity_summary.csv"


# ============================================================
# IO
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ============================================================
# Helpers
# ============================================================

DEFAULT_METRIC_MAP = {
    "unsafe_excess": "total_unsafe_excess_mean",
    "blocked_arrivals": "total_blocked_arrivals_mean",
    "max_utilization": "max_utilization_ratio_mean",
    "unsafe_rows": "num_unsafe_rows_mean",
    "overflow_excess": "total_overflow_excess_mean",
    "surge_gap": "total_surge_gap_mean",
}

LOWER_IS_BETTER_METRICS = set(DEFAULT_METRIC_MAP.values())


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        return str(x)
    except Exception:
        return default


def metric_direction_text(metric_col: str, diff_selected_minus_benchmark: float) -> str:
    if pd.isna(diff_selected_minus_benchmark):
        return "unknown"

    if metric_col in LOWER_IS_BETTER_METRICS:
        if diff_selected_minus_benchmark < 0:
            return "improves"
        if diff_selected_minus_benchmark > 0:
            return "worsens"
        return "does_not_change"

    if diff_selected_minus_benchmark > 0:
        return "improves"
    if diff_selected_minus_benchmark < 0:
        return "worsens"
    return "does_not_change"


def percent_change(selected: float, benchmark: float) -> float:
    if pd.isna(selected) or pd.isna(benchmark):
        return np.nan
    if abs(benchmark) < 1e-12:
        return np.nan
    return 100.0 * (selected - benchmark) / benchmark


def choose_existing_metrics(df: pd.DataFrame) -> dict[str, str]:
    return {k: v for k, v in DEFAULT_METRIC_MAP.items() if v in df.columns}


# ============================================================
# Row selection
# ============================================================

def filter_single_row(
    df: pd.DataFrame,
    *,
    scenario: str | None = None,
    design_name: str | None = None,
    policy_name: str | None = None,
) -> pd.DataFrame:
    out = df.copy()

    if scenario is not None and "scenario" in out.columns:
        out = out[out["scenario"] == scenario]

    if design_name is not None and "design_name" in out.columns:
        out = out[out["design_name"] == design_name]

    if policy_name is not None and "policy_name" in out.columns:
        out = out[out["policy_name"] == policy_name]

    return out.reset_index(drop=True)


def get_single_row(
    df: pd.DataFrame,
    *,
    scenario: str | None = None,
    design_name: str | None = None,
    policy_name: str | None = None,
) -> dict[str, Any]:
    rows = filter_single_row(
        df,
        scenario=scenario,
        design_name=design_name,
        policy_name=policy_name,
    )
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


# ============================================================
# Core counterfactual comparison
# ============================================================

def compare_rows(
    selected_row: dict[str, Any] | pd.Series,
    benchmark_row: dict[str, Any] | pd.Series,
    *,
    metric_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if metric_map is None:
        keys = set()
        if isinstance(selected_row, dict):
            keys.update(selected_row.keys())
        else:
            keys.update(selected_row.index.tolist())

        if isinstance(benchmark_row, dict):
            keys.update(benchmark_row.keys())
        else:
            keys.update(benchmark_row.index.tolist())

        metric_map = {k: v for k, v in DEFAULT_METRIC_MAP.items() if v in keys}

    rows = []

    for metric_label, metric_col in metric_map.items():
        s = safe_float(selected_row.get(metric_col) if isinstance(selected_row, dict) else selected_row[metric_col])
        b = safe_float(benchmark_row.get(metric_col) if isinstance(benchmark_row, dict) else benchmark_row[metric_col])

        diff = s - b if pd.notna(s) and pd.notna(b) else np.nan
        pct = percent_change(s, b)
        direction = metric_direction_text(metric_col, diff)

        rows.append(
            {
                "metric_label": metric_label,
                "metric_col": metric_col,
                "selected_value": s,
                "benchmark_value": b,
                "difference_selected_minus_benchmark": diff,
                "percent_change_selected_vs_benchmark": pct,
                "direction": direction,
            }
        )

    out = pd.DataFrame(rows)
    return out


def compare_policy_within_scenario(
    summary_df: pd.DataFrame,
    *,
    scenario: str,
    selected_policy: str,
    benchmark_policy: str,
) -> pd.DataFrame:
    selected_row = get_single_row(
        summary_df,
        scenario=scenario,
        policy_name=selected_policy,
    )
    benchmark_row = get_single_row(
        summary_df,
        scenario=scenario,
        policy_name=benchmark_policy,
    )

    if not selected_row or not benchmark_row:
        return pd.DataFrame()

    out = compare_rows(selected_row, benchmark_row)
    out["scenario"] = scenario
    out["selected_policy"] = selected_policy
    out["benchmark_policy"] = benchmark_policy
    return out


def compare_design_within_policy(
    sensitivity_df: pd.DataFrame,
    *,
    policy_name: str,
    selected_design: str,
    benchmark_design: str,
) -> pd.DataFrame:
    selected_row = get_single_row(
        sensitivity_df,
        design_name=selected_design,
        policy_name=policy_name,
    )
    benchmark_row = get_single_row(
        sensitivity_df,
        design_name=benchmark_design,
        policy_name=policy_name,
    )

    if not selected_row or not benchmark_row:
        return pd.DataFrame()

    out = compare_rows(selected_row, benchmark_row)
    out["policy_name"] = policy_name
    out["selected_design"] = selected_design
    out["benchmark_design"] = benchmark_design
    return out


# ============================================================
# Narrative rendering
# ============================================================

def metric_sentence(row: pd.Series) -> str:
    metric = safe_str(row.get("metric_label"))
    diff = safe_float(row.get("difference_selected_minus_benchmark"))
    pct = safe_float(row.get("percent_change_selected_vs_benchmark"))
    direction = safe_str(row.get("direction"), "unknown")

    if pd.isna(diff):
        return f"{metric} comparison is unavailable."

    abs_diff = abs(diff)

    if direction == "improves":
        if pd.notna(pct):
            return f"{metric} improves by {abs_diff:.2f} ({abs(pct):.1f}%)."
        return f"{metric} improves by {abs_diff:.2f}."

    if direction == "worsens":
        if pd.notna(pct):
            return f"{metric} worsens by {abs_diff:.2f} ({abs(pct):.1f}%)."
        return f"{metric} worsens by {abs_diff:.2f}."

    return f"{metric} does not materially change."


def build_counterfactual_summary(
    comparison_df: pd.DataFrame,
    *,
    headline_metrics: list[str] | None = None,
) -> dict[str, Any]:
    if comparison_df.empty:
        return {
            "headline": "No counterfactual comparison was available.",
            "sentences": [],
            "comparison_table": comparison_df,
        }

    if headline_metrics is None:
        headline_metrics = ["unsafe_excess", "blocked_arrivals", "max_utilization"]

    work = comparison_df.copy()
    work = work[work["metric_label"].isin(headline_metrics)].copy()

    sentences = [metric_sentence(row) for _, row in work.iterrows()]

    headline_parts = []
    for metric in ["unsafe_excess", "blocked_arrivals"]:
        sdf = work[work["metric_label"] == metric]
        if not sdf.empty:
            headline_parts.append(metric_sentence(sdf.iloc[0]))

    headline = " ".join(headline_parts) if headline_parts else "Counterfactual comparison generated."

    return {
        "headline": headline,
        "sentences": sentences,
        "comparison_table": comparison_df,
    }


# ============================================================
# Batch builders for paper/reporting
# ============================================================

def build_policy_counterfactual_table(
    summary_df: pd.DataFrame,
    *,
    policy_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    if summary_df.empty or "scenario" not in summary_df.columns:
        return pd.DataFrame()

    if policy_pairs is None:
        policy_pairs = [
            ("regime_robust_optimized_network", "optimized_network"),
            ("regime_robust_optimized_network", "robust_optimized_network"),
            ("robust_optimized_network", "optimized_network"),
        ]

    rows = []
    scenarios = sorted(summary_df["scenario"].dropna().unique().tolist())

    for scenario in scenarios:
        for selected_policy, benchmark_policy in policy_pairs:
            comp = compare_policy_within_scenario(
                summary_df,
                scenario=scenario,
                selected_policy=selected_policy,
                benchmark_policy=benchmark_policy,
            )
            if not comp.empty:
                rows.append(comp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_sensitivity_counterfactual_table(
    sensitivity_df: pd.DataFrame,
    *,
    policy_name: str = "regime_robust_optimized_network",
    benchmark_design: str = "base",
) -> pd.DataFrame:
    if sensitivity_df.empty or "design_name" not in sensitivity_df.columns:
        return pd.DataFrame()

    rows = []
    designs = sorted(sensitivity_df["design_name"].dropna().unique().tolist())

    for design in designs:
        if design == benchmark_design:
            continue

        comp = compare_design_within_policy(
            sensitivity_df,
            policy_name=policy_name,
            selected_design=design,
            benchmark_design=benchmark_design,
        )
        if not comp.empty:
            rows.append(comp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Example main
# ============================================================

def main() -> None:
    regime_summary = safe_read_csv(REGIME_SUMMARY_PATH)
    sensitivity_summary = safe_read_csv(SENSITIVITY_SUMMARY_PATH)

    print("\nCOUNTERFACTUAL ENGINE")
    print("=" * 70)

    # Example 1: policy comparison within a scenario
    policy_comp = compare_policy_within_scenario(
        regime_summary,
        scenario="baseline",
        selected_policy="regime_robust_optimized_network",
        benchmark_policy="optimized_network",
    )

    print("\nPolicy counterfactual: baseline, regime robust vs optimized")
    print(policy_comp)

    policy_summary = build_counterfactual_summary(policy_comp)
    print("\nPolicy summary headline:")
    print(policy_summary["headline"])

    # Example 2: design comparison within a policy
    design_comp = compare_design_within_policy(
        sensitivity_summary,
        policy_name="regime_robust_optimized_network",
        selected_design="icu_minus_20",
        benchmark_design="base",
    )

    print("\nSensitivity counterfactual: ICU -20 vs base under regime robust")
    print(design_comp)

    design_summary = build_counterfactual_summary(design_comp)
    print("\nSensitivity summary headline:")
    print(design_summary["headline"])

    # Save batch outputs if possible
    policy_batch = build_policy_counterfactual_table(regime_summary)
    sensitivity_batch = build_sensitivity_counterfactual_table(sensitivity_summary)

    policy_path = TABLES_DIR / "counterfactual_policy_comparisons.csv"
    sensitivity_path = TABLES_DIR / "counterfactual_sensitivity_comparisons.csv"

    if not policy_batch.empty:
        policy_batch.to_csv(policy_path, index=False)
    if not sensitivity_batch.empty:
        sensitivity_batch.to_csv(sensitivity_path, index=False)

    print("\nSaved:")
    print(policy_path if policy_path.exists() else f"{policy_path} [not created]")
    print(sensitivity_path if sensitivity_path.exists() else f"{sensitivity_path} [not created]")


if __name__ == "__main__":
    main()