from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = AI_DATA_DIR / "forecast_dataset_combined.csv"

OUT_DIST_PATH = TABLES_DIR / "regime_distribution_audit.csv"
OUT_TRANSITION_PATH = TABLES_DIR / "regime_transition_matrix.csv"
OUT_SCENARIO_PATH = TABLES_DIR / "regime_distribution_by_scenario.csv"
OUT_POLICY_PATH = TABLES_DIR / "regime_distribution_by_policy.csv"
OUT_SUMMARY_JSON = TABLES_DIR / "regime_audit_summary.json"

FIG_DIST = FIGURES_DIR / "regime_distribution_audit.png"
FIG_TRANSITION = FIGURES_DIR / "regime_transition_heatmap.png"


# ============================================================
# Helpers
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def normalize_counts(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    total = counts.sum()

    df = pd.DataFrame({
        "label": counts.index.astype(str),
        "count": counts.values,
    })
    df["percentage"] = df["count"] / total

    return df.sort_values("count", ascending=False).reset_index(drop=True)


# ============================================================
# Core Audit
# ============================================================

def compute_overall_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "next_regime_label" not in df.columns:
        raise ValueError("next_regime_label column missing")

    return normalize_counts(df["next_regime_label"])


def compute_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if "current_regime" not in df.columns or "next_regime_label" not in df.columns:
        return pd.DataFrame()

    pivot = pd.crosstab(
        df["current_regime"],
        df["next_regime_label"],
        normalize="index",
    )

    return pivot


def compute_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    rows = []

    for g, sub in df.groupby(group_col):
        dist = normalize_counts(sub["next_regime_label"])

        for _, r in dist.iterrows():
            rows.append({
                group_col: g,
                "regime": r["label"],
                "count": r["count"],
                "percentage": r["percentage"],
            })

    return pd.DataFrame(rows)


# ============================================================
# Visualization
# ============================================================

def plot_distribution(dist_df: pd.DataFrame) -> None:
    plt.figure()

    plt.bar(dist_df["label"], dist_df["percentage"])
    plt.xlabel("Regime")
    plt.ylabel("Percentage")
    plt.title("Next Regime Distribution")

    plt.tight_layout()
    plt.savefig(FIG_DIST)
    plt.close()


def plot_transition_heatmap(matrix: pd.DataFrame) -> None:
    if matrix.empty:
        return

    plt.figure()

    plt.imshow(matrix.values)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45)
    plt.yticks(range(len(matrix.index)), matrix.index)

    plt.colorbar(label="Transition Probability")
    plt.title("Regime Transition Matrix")

    plt.tight_layout()
    plt.savefig(FIG_TRANSITION)
    plt.close()


# ============================================================
# Diagnostics Logic
# ============================================================

def diagnose_distribution(dist_df: pd.DataFrame) -> dict:
    summary = {}

    if dist_df.empty:
        return {"status": "no_data"}

    crisis_row = dist_df[dist_df["label"].str.lower() == "crisis"]

    if not crisis_row.empty:
        crisis_pct = float(crisis_row.iloc[0]["percentage"])
    else:
        crisis_pct = 0.0

    summary["crisis_percentage"] = crisis_pct

    if crisis_pct >= 0.80:
        summary["assessment"] = "extreme_crisis_dominance"
        summary["risk_to_paper"] = "HIGH"
        summary["interpretation"] = (
            "System operates almost always in crisis regime. "
            "Regime-aware modeling may not be meaningfully exercised."
        )

    elif crisis_pct >= 0.60:
        summary["assessment"] = "high_crisis_bias"
        summary["risk_to_paper"] = "MEDIUM"
        summary["interpretation"] = (
            "Crisis dominates but other regimes exist. "
            "Need scenario-wise validation."
        )

    else:
        summary["assessment"] = "balanced_or_reasonable"
        summary["risk_to_paper"] = "LOW"
        summary["interpretation"] = (
            "Regime distribution is sufficiently diverse for testing."
        )

    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nREGIME DISTRIBUTION AUDIT")
    print("=" * 70)

    df = safe_read_csv(DATA_PATH)

    overall_dist = compute_overall_distribution(df)
    transition_matrix = compute_transition_matrix(df)
    scenario_dist = compute_by_group(df, "scenario")
    policy_dist = compute_by_group(df, "policy_name")

    # Save tables
    overall_dist.to_csv(OUT_DIST_PATH, index=False)
    transition_matrix.to_csv(OUT_TRANSITION_PATH)
    scenario_dist.to_csv(OUT_SCENARIO_PATH, index=False)
    policy_dist.to_csv(OUT_POLICY_PATH, index=False)

    # Plots
    plot_distribution(overall_dist)
    plot_transition_heatmap(transition_matrix)

    # Diagnostics
    summary = diagnose_distribution(overall_dist)

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    # Print
    print("\n=== Overall Distribution ===")
    print(overall_dist.to_string(index=False))

    print("\n=== Transition Matrix ===")
    print(transition_matrix.to_string())

    print("\n=== Scenario Distribution ===")
    print(scenario_dist.head(20).to_string(index=False))

    print("\n=== Policy Distribution ===")
    print(policy_dist.head(20).to_string(index=False))

    print("\n=== Diagnostic Summary ===")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(OUT_DIST_PATH)
    print(OUT_TRANSITION_PATH)
    print(OUT_SCENARIO_PATH)
    print(OUT_POLICY_PATH)
    print(OUT_SUMMARY_JSON)
    print(FIG_DIST)
    print(FIG_TRANSITION)


if __name__ == "__main__":
    main()