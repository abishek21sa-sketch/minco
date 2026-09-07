from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"
FIGURES_DIR = RESULTS_DIR / "figures"

AI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = AI_DATA_DIR / "forecast_dataset_combined.csv"
TARGET_COL = "target_next_total_unsafe_excess"

SUMMARY_PATH = AI_REPORTS_DIR / "unsafe_excess_feature_diagnostics_summary.json"
CORRELATION_PATH = AI_REPORTS_DIR / "unsafe_excess_feature_correlations.csv"
GROUP_SUMMARY_PATH = AI_REPORTS_DIR / "unsafe_excess_group_summary.csv"
ZERO_INFLATION_PATH = AI_REPORTS_DIR / "unsafe_excess_zero_inflation.csv"

TARGET_HIST_FIG = FIGURES_DIR / "unsafe_excess_target_distribution.png"
TARGET_BY_SCENARIO_FIG = FIGURES_DIR / "unsafe_excess_by_scenario.png"
TARGET_BY_POLICY_FIG = FIGURES_DIR / "unsafe_excess_by_policy.png"
TOP_CORR_FIG = FIGURES_DIR / "unsafe_excess_top_correlations.png"


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c == TARGET_COL:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
    return cols


def build_target_summary(df: pd.DataFrame) -> dict:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

    if y.empty:
        return {}

    return {
        "n": int(len(y)),
        "mean": float(y.mean()),
        "std": float(y.std(ddof=1)),
        "min": float(y.min()),
        "p05": float(y.quantile(0.05)),
        "p25": float(y.quantile(0.25)),
        "median": float(y.median()),
        "p75": float(y.quantile(0.75)),
        "p90": float(y.quantile(0.90)),
        "p95": float(y.quantile(0.95)),
        "max": float(y.max()),
        "zero_count": int((y == 0).sum()),
        "zero_share": float((y == 0).mean()),
        "positive_count": int((y > 0).sum()),
        "positive_share": float((y > 0).mean()),
        "high_gt_5_count": int((y > 5).sum()),
        "high_gt_5_share": float((y > 5).mean()),
        "critical_gt_10_count": int((y > 10).sum()),
        "critical_gt_10_share": float((y > 10).mean()),
    }


def build_zero_inflation_table(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    rows = []

    group_cols = [
        "dataset_source",
        "scenario",
        "policy_name",
        "current_regime",
        "design_name",
    ]

    for group_col in group_cols:
        if group_col not in df.columns:
            continue

        tmp = df[[group_col]].copy()
        tmp[TARGET_COL] = y

        grouped = (
            tmp.dropna(subset=[TARGET_COL])
            .groupby(group_col, as_index=False)
            .agg(
                n=(TARGET_COL, "count"),
                mean_target=(TARGET_COL, "mean"),
                median_target=(TARGET_COL, "median"),
                std_target=(TARGET_COL, "std"),
                zero_share=(TARGET_COL, lambda s: float((s == 0).mean())),
                positive_share=(TARGET_COL, lambda s: float((s > 0).mean())),
                high_gt_5_share=(TARGET_COL, lambda s: float((s > 5).mean())),
                critical_gt_10_share=(TARGET_COL, lambda s: float((s > 10).mean())),
            )
        )

        grouped.insert(0, "group_variable", group_col)
        grouped = grouped.rename(columns={group_col: "group_value"})
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    work = df.copy()
    work[TARGET_COL] = y
    work = work.dropna(subset=[TARGET_COL]).copy()

    group_specs = [
        ["scenario"],
        ["policy_name"],
        ["current_regime"],
        ["dataset_source"],
        ["scenario", "policy_name"],
        ["current_regime", "policy_name"],
    ]

    rows = []

    for group_cols in group_specs:
        if not all(c in work.columns for c in group_cols):
            continue

        grouped = (
            work.groupby(group_cols, as_index=False)
            .agg(
                n=(TARGET_COL, "count"),
                mean_target=(TARGET_COL, "mean"),
                median_target=(TARGET_COL, "median"),
                std_target=(TARGET_COL, "std"),
                p75_target=(TARGET_COL, lambda s: float(s.quantile(0.75))),
                p90_target=(TARGET_COL, lambda s: float(s.quantile(0.90))),
                max_target=(TARGET_COL, "max"),
                positive_share=(TARGET_COL, lambda s: float((s > 0).mean())),
                high_gt_5_share=(TARGET_COL, lambda s: float((s > 5).mean())),
                critical_gt_10_share=(TARGET_COL, lambda s: float((s > 10).mean())),
            )
        )

        grouped.insert(0, "grouping", " + ".join(group_cols))
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True, sort=False)


def build_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    rows = []
    for col in numeric_columns(df):
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.notna() & y.notna()

        if valid.sum() < 10:
            continue

        xv = x[valid]
        yv = y[valid]

        if xv.std(ddof=1) <= 1e-12 or yv.std(ddof=1) <= 1e-12:
            continue

        corr = float(xv.corr(yv))

        rows.append(
            {
                "feature": col,
                "n": int(valid.sum()),
                "pearson_corr": corr,
                "abs_corr": abs(corr),
                "feature_mean": float(xv.mean()),
                "feature_std": float(xv.std(ddof=1)),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )


def plot_target_distribution(df: pd.DataFrame) -> None:
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()

    if y.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=40)
    plt.xlabel("Next unsafe excess")
    plt.ylabel("Frequency")
    plt.title("Distribution of Next-Period Unsafe Excess")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TARGET_HIST_FIG, dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_bar(
    df: pd.DataFrame,
    group_col: str,
    path: Path,
    title: str,
    top_n: int | None = None,
) -> None:
    if group_col not in df.columns:
        return

    work = df[[group_col, TARGET_COL]].copy()
    work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce")
    work = work.dropna(subset=[TARGET_COL])

    if work.empty:
        return

    grouped = (
        work.groupby(group_col, as_index=False)
        .agg(mean_target=(TARGET_COL, "mean"))
        .sort_values("mean_target", ascending=False)
    )

    if top_n is not None:
        grouped = grouped.head(top_n)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(grouped)), grouped["mean_target"])
    plt.xticks(range(len(grouped)), grouped[group_col].astype(str), rotation=25, ha="right")
    plt.ylabel("Mean next unsafe excess")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_correlations(corr_df: pd.DataFrame) -> None:
    if corr_df.empty:
        return

    top = corr_df.head(20).copy()
    top = top.sort_values("abs_corr", ascending=True)

    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["pearson_corr"])
    plt.xlabel("Pearson correlation with next unsafe excess")
    plt.ylabel("Feature")
    plt.title("Top Correlated Features for Next Unsafe Excess")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(TOP_CORR_FIG, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("\nUNSAFE EXCESS FEATURE DIAGNOSTICS")
    print("=" * 70)

    df = safe_read_csv(INPUT_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()

    target_summary = build_target_summary(df)
    zero_table = build_zero_inflation_table(df)
    group_summary = build_group_summary(df)
    corr_df = build_correlation_table(df)

    SUMMARY_PATH.write_text(json.dumps(target_summary, indent=2), encoding="utf-8")
    zero_table.to_csv(ZERO_INFLATION_PATH, index=False)
    group_summary.to_csv(GROUP_SUMMARY_PATH, index=False)
    corr_df.to_csv(CORRELATION_PATH, index=False)

    plot_target_distribution(df)
    plot_group_bar(
        df,
        group_col="scenario",
        path=TARGET_BY_SCENARIO_FIG,
        title="Mean Next Unsafe Excess by Scenario",
    )
    plot_group_bar(
        df,
        group_col="policy_name",
        path=TARGET_BY_POLICY_FIG,
        title="Mean Next Unsafe Excess by Policy",
    )
    plot_top_correlations(corr_df)

    print("\n=== Target summary ===")
    print(json.dumps(target_summary, indent=2))

    print("\n=== Top correlations ===")
    if corr_df.empty:
        print("No correlations computed.")
    else:
        print(corr_df.head(20).to_string(index=False))

    print("\n=== Zero inflation preview ===")
    if zero_table.empty:
        print("No zero-inflation table.")
    else:
        print(zero_table.head(20).to_string(index=False))

    print("\nSaved:")
    print(SUMMARY_PATH)
    print(CORRELATION_PATH)
    print(GROUP_SUMMARY_PATH)
    print(ZERO_INFLATION_PATH)
    print(TARGET_HIST_FIG)
    print(TARGET_BY_SCENARIO_FIG)
    print(TARGET_BY_POLICY_FIG)
    print(TOP_CORR_FIG)


if __name__ == "__main__":
    main()