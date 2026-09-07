from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = AI_DATA_DIR / "forecast_dataset_combined.csv"

OUT_ROW_LEVEL_PATH = TABLES_DIR / "regime_recalibration_row_level.csv"
OUT_DISTRIBUTION_PATH = TABLES_DIR / "regime_recalibration_distribution.csv"
OUT_SCENARIO_PATH = TABLES_DIR / "regime_recalibration_by_scenario.csv"
OUT_POLICY_PATH = TABLES_DIR / "regime_recalibration_by_policy.csv"
OUT_TRANSITION_PATH = TABLES_DIR / "regime_recalibration_transition_matrix.csv"
OUT_RECOMMENDATION_PATH = TABLES_DIR / "regime_recalibration_recommendations.csv"
OUT_SUMMARY_JSON = TABLES_DIR / "regime_recalibration_summary.json"

FIG_DIST = FIGURES_DIR / "regime_recalibration_distribution.png"
FIG_SCENARIO = FIGURES_DIR / "regime_recalibration_by_scenario.png"


# ============================================================
# Calibration thresholds
# ============================================================

THRESHOLDS = {
    "normal": {
        "max_utilization_upper": 0.90,
        "unsafe_excess_upper": 1.00,
        "blocked_arrivals_upper": 3.00,
        "overflow_excess_upper": 0.50,
    },
    "surge": {
        "max_utilization_upper": 1.05,
        "unsafe_excess_upper": 6.00,
        "blocked_arrivals_upper": 10.00,
        "overflow_excess_upper": 3.00,
    },
    "crisis": {
        "max_utilization_lower": 1.05,
        "unsafe_excess_lower": 6.00,
        "blocked_arrivals_lower": 10.00,
        "overflow_excess_lower": 3.00,
    },
}


SAFE_NUMERIC_COLS = [
    "max_utilization_ratio",
    "total_unsafe_excess",
    "total_blocked_arrivals",
    "total_overflow_excess",
    "total_surge_gap",
    "num_unsafe_rows",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in SAFE_NUMERIC_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


# ============================================================
# Rule-based recalibrated labels
# ============================================================

def classify_operational_regime(row: pd.Series) -> str:
    util = safe_float(row.get("max_utilization_ratio"))
    unsafe = safe_float(row.get("total_unsafe_excess"))
    blocked = safe_float(row.get("total_blocked_arrivals"))
    overflow = safe_float(row.get("total_overflow_excess"))
    unsafe_rows = safe_float(row.get("num_unsafe_rows"))

    crisis_score = 0
    surge_score = 0

    if util >= THRESHOLDS["crisis"]["max_utilization_lower"]:
        crisis_score += 2
    elif util >= THRESHOLDS["normal"]["max_utilization_upper"]:
        surge_score += 2

    if unsafe >= THRESHOLDS["crisis"]["unsafe_excess_lower"]:
        crisis_score += 2
    elif unsafe >= THRESHOLDS["normal"]["unsafe_excess_upper"]:
        surge_score += 1

    if blocked >= THRESHOLDS["crisis"]["blocked_arrivals_lower"]:
        crisis_score += 1
    elif blocked >= THRESHOLDS["normal"]["blocked_arrivals_upper"]:
        surge_score += 1

    if overflow >= THRESHOLDS["crisis"]["overflow_excess_lower"]:
        crisis_score += 1
    elif overflow >= THRESHOLDS["normal"]["overflow_excess_upper"]:
        surge_score += 1

    if unsafe_rows >= 3:
        crisis_score += 1
    elif unsafe_rows >= 1:
        surge_score += 1

    if crisis_score >= 2:
        return "crisis"

    if surge_score >= 2:
        return "surge"

    return "normal"


def compute_regime_score(row: pd.Series) -> float:
    util = safe_float(row.get("max_utilization_ratio"))
    unsafe = safe_float(row.get("total_unsafe_excess"))
    blocked = safe_float(row.get("total_blocked_arrivals"))
    overflow = safe_float(row.get("total_overflow_excess"))
    surge_gap = safe_float(row.get("total_surge_gap"))
    unsafe_rows = safe_float(row.get("num_unsafe_rows"))

    score = (
        3.0 * max(util - 0.85, 0.0)
        + 0.20 * unsafe
        + 0.08 * blocked
        + 0.25 * overflow
        + 0.10 * surge_gap
        + 0.25 * unsafe_rows
    )

    return float(score)


def score_based_label(score: float) -> str:
    if score < 0.75:
        return "normal"
    if score < 2.00:
        return "surge"
    return "crisis"


def add_recalibrated_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_numeric(df)

    out["operational_regime_score"] = out.apply(compute_regime_score, axis=1)
    out["calibrated_regime_label_rule"] = out.apply(classify_operational_regime, axis=1)
    out["calibrated_regime_label_score"] = out["operational_regime_score"].apply(score_based_label)

    if "next_regime_label" in out.columns:
        out["original_next_regime_label"] = out["next_regime_label"].astype(str)
        out["label_changed_rule"] = (
            out["original_next_regime_label"].astype(str)
            != out["calibrated_regime_label_rule"].astype(str)
        )
        out["label_changed_score"] = (
            out["original_next_regime_label"].astype(str)
            != out["calibrated_regime_label_score"].astype(str)
        )

    return out


# ============================================================
# Summaries
# ============================================================

def distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()

    counts = df[col].astype(str).value_counts(dropna=False)
    total = counts.sum()

    out = pd.DataFrame(
        {
            "regime": counts.index,
            "count": counts.values,
        }
    )
    out["percentage"] = out["count"] / total
    out["label_source"] = col

    return out.sort_values("regime").reset_index(drop=True)


def grouped_distribution(df: pd.DataFrame, group_col: str, label_col: str) -> pd.DataFrame:
    if group_col not in df.columns or label_col not in df.columns:
        return pd.DataFrame()

    rows = []

    for group, sub in df.groupby(group_col):
        dist = distribution(sub, label_col)
        for _, r in dist.iterrows():
            rows.append(
                {
                    group_col: group,
                    "label_source": label_col,
                    "regime": r["regime"],
                    "count": int(r["count"]),
                    "percentage": float(r["percentage"]),
                }
            )

    return pd.DataFrame(rows)


def transition_matrix(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if "current_regime" not in df.columns or label_col not in df.columns:
        return pd.DataFrame()

    return pd.crosstab(
        df["current_regime"].astype(str),
        df[label_col].astype(str),
        normalize="index",
    )


def compare_original_vs_calibrated(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label_col in [
        "original_next_regime_label",
        "calibrated_regime_label_rule",
        "calibrated_regime_label_score",
    ]:
        if label_col in df.columns:
            rows.append(distribution(df, label_col))

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_recommendations(df: pd.DataFrame, dist_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    original = dist_df[dist_df["label_source"] == "original_next_regime_label"].copy()
    rule = dist_df[dist_df["label_source"] == "calibrated_regime_label_rule"].copy()
    score = dist_df[dist_df["label_source"] == "calibrated_regime_label_score"].copy()

    def crisis_pct(sub: pd.DataFrame) -> float:
        row = sub[sub["regime"].str.lower() == "crisis"]
        if row.empty:
            return 0.0
        return float(row.iloc[0]["percentage"])

    original_crisis = crisis_pct(original)
    rule_crisis = crisis_pct(rule)
    score_crisis = crisis_pct(score)

    rows.append(
        {
            "issue": "original_crisis_dominance",
            "finding": f"Original next_regime_label crisis share = {original_crisis:.3f}",
            "recommendation": (
                "Do not claim broad regime-switching validation from original labels alone "
                "because crisis dominates the dataset."
            ),
            "priority": "critical",
        }
    )

    rows.append(
        {
            "issue": "rule_based_recalibration",
            "finding": f"Rule-based calibrated crisis share = {rule_crisis:.3f}",
            "recommendation": (
                "Use calibrated_regime_label_rule as a diagnostic label to test whether "
                "operational metrics support a more balanced regime definition."
            ),
            "priority": "high",
        }
    )

    rows.append(
        {
            "issue": "score_based_recalibration",
            "finding": f"Score-based calibrated crisis share = {score_crisis:.3f}",
            "recommendation": (
                "Use calibrated_regime_label_score as an alternative continuous-stress "
                "calibration. If it is substantially more balanced, rerun regime analysis "
                "using calibrated labels or rerun simulation with softer stress parameters."
            ),
            "priority": "high",
        }
    )

    if rule_crisis < 0.70 or score_crisis < 0.70:
        decision = (
            "Recalibrated labels reduce crisis dominance. Recommended next step: build a "
            "calibrated regime suite and rerun regime-aware policy analysis using the "
            "calibrated regime label."
        )
    else:
        decision = (
            "Even calibrated labels remain crisis-heavy. Recommended next step: adjust the "
            "simulation demand/capacity parameters to create normal, surge, and crisis "
            "conditions explicitly."
        )

    rows.append(
        {
            "issue": "next_action",
            "finding": "Calibration decision generated from crisis shares.",
            "recommendation": decision,
            "priority": "critical",
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

def plot_distribution(dist_df: pd.DataFrame) -> None:
    if dist_df.empty:
        return

    plot_df = dist_df.copy()
    sources = plot_df["label_source"].unique().tolist()
    regimes = sorted(plot_df["regime"].unique().tolist())

    x = np.arange(len(regimes))
    width = 0.8 / max(len(sources), 1)

    plt.figure(figsize=(10, 6))

    for i, source in enumerate(sources):
        sub = plot_df[plot_df["label_source"] == source].set_index("regime")
        vals = [float(sub.loc[r, "percentage"]) if r in sub.index else 0.0 for r in regimes]
        plt.bar(x + i * width, vals, width=width, label=source)

    plt.xticks(x + width * (len(sources) - 1) / 2, regimes)
    plt.ylabel("Percentage")
    plt.title("Original vs Recalibrated Regime Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIST, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scenario_distribution(scenario_df: pd.DataFrame) -> None:
    if scenario_df.empty:
        return

    # Plot score-based labels only for readability
    plot_df = scenario_df[
        scenario_df["label_source"] == "calibrated_regime_label_score"
    ].copy()

    if plot_df.empty:
        return

    pivot = plot_df.pivot_table(
        index="scenario",
        columns="regime",
        values="percentage",
        fill_value=0.0,
    )

    pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
    plt.ylabel("Percentage")
    plt.title("Score-Based Calibrated Regime Distribution by Scenario")
    plt.tight_layout()
    plt.savefig(FIG_SCENARIO, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nREGIME RECALIBRATION PLAN")
    print("=" * 70)

    raw = safe_read_csv(DATA_PATH)
    df = add_recalibrated_labels(raw)

    dist_df = compare_original_vs_calibrated(df)

    scenario_frames = []
    policy_frames = []

    for label_col in [
        "original_next_regime_label",
        "calibrated_regime_label_rule",
        "calibrated_regime_label_score",
    ]:
        scenario_frames.append(grouped_distribution(df, "scenario", label_col))
        policy_frames.append(grouped_distribution(df, "policy_name", label_col))

    scenario_df = pd.concat(
        [x for x in scenario_frames if not x.empty],
        ignore_index=True,
    ) if scenario_frames else pd.DataFrame()

    policy_df = pd.concat(
        [x for x in policy_frames if not x.empty],
        ignore_index=True,
    ) if policy_frames else pd.DataFrame()

    transition_rule = transition_matrix(df, "calibrated_regime_label_rule")
    transition_score = transition_matrix(df, "calibrated_regime_label_score")

    recommendations = build_recommendations(df, dist_df)

    # Save row-level diagnostic, but not every original column if huge
    keep_patterns = [
        "dataset_source",
        "scenario",
        "policy_name",
        "time_index",
        "current_regime",
        "next_regime_label",
        "original_next_regime_label",
        "calibrated_regime_label_rule",
        "calibrated_regime_label_score",
        "operational_regime_score",
        "label_changed_rule",
        "label_changed_score",
        "total_unsafe_excess",
        "total_blocked_arrivals",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_overflow_excess",
        "total_surge_gap",
        "lag",
        "rollmean",
        "rollmax",
        "flag",
    ]

    cols_to_save = [
        c for c in df.columns
        if any(pattern in c for pattern in keep_patterns)
    ]

    df[cols_to_save].to_csv(OUT_ROW_LEVEL_PATH, index=False)
    dist_df.to_csv(OUT_DISTRIBUTION_PATH, index=False)
    scenario_df.to_csv(OUT_SCENARIO_PATH, index=False)
    policy_df.to_csv(OUT_POLICY_PATH, index=False)

    # Save both transition matrices in long form
    transition_rows = []
    for name, mat in [
        ("rule", transition_rule),
        ("score", transition_score),
    ]:
        if mat.empty:
            continue
        tmp = mat.reset_index().melt(
            id_vars="current_regime",
            var_name="next_regime",
            value_name="transition_probability",
        )
        tmp["calibration_method"] = name
        transition_rows.append(tmp)

    transition_out = pd.concat(transition_rows, ignore_index=True) if transition_rows else pd.DataFrame()
    transition_out.to_csv(OUT_TRANSITION_PATH, index=False)

    recommendations.to_csv(OUT_RECOMMENDATION_PATH, index=False)

    plot_distribution(dist_df)
    plot_scenario_distribution(scenario_df)

    summary = {
        "n_rows": int(len(df)),
        "thresholds": THRESHOLDS,
        "distribution": dist_df.to_dict(orient="records"),
        "recommendations": recommendations.to_dict(orient="records"),
        "outputs": {
            "row_level": str(OUT_ROW_LEVEL_PATH),
            "distribution": str(OUT_DISTRIBUTION_PATH),
            "scenario": str(OUT_SCENARIO_PATH),
            "policy": str(OUT_POLICY_PATH),
            "transition": str(OUT_TRANSITION_PATH),
            "recommendations": str(OUT_RECOMMENDATION_PATH),
            "figure_distribution": str(FIG_DIST),
            "figure_scenario": str(FIG_SCENARIO),
        },
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Distribution Comparison ===")
    print(dist_df.to_string(index=False))

    print("\n=== Scenario Distribution Preview ===")
    if scenario_df.empty:
        print("No scenario distribution available.")
    else:
        print(scenario_df.head(30).to_string(index=False))

    print("\n=== Recommendations ===")
    print(recommendations.to_string(index=False))

    print("\nSaved:")
    print(OUT_ROW_LEVEL_PATH)
    print(OUT_DISTRIBUTION_PATH)
    print(OUT_SCENARIO_PATH)
    print(OUT_POLICY_PATH)
    print(OUT_TRANSITION_PATH)
    print(OUT_RECOMMENDATION_PATH)
    print(OUT_SUMMARY_JSON)
    print(FIG_DIST)
    print(FIG_SCENARIO)


if __name__ == "__main__":
    main()