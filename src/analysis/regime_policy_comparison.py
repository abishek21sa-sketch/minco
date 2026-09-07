from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = TABLES_DIR / "regime_suite_summary.csv"
REPL_PATH = TABLES_DIR / "regime_suite_replications.csv"
DELTA_PATH = TABLES_DIR / "regime_suite_delta.csv"


# ============================================================
# IO
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ============================================================
# Metric helpers
# ============================================================

MINIMIZE_METRICS = [
    "total_unsafe_excess_mean",
    "total_blocked_arrivals_mean",
    "max_utilization_ratio_mean",
    "total_overflow_excess_mean",
    "num_unsafe_rows_mean",
    "total_surge_gap_mean",
]

PRETTY_NAMES = {
    "total_unsafe_excess_mean": "unsafe_excess",
    "total_blocked_arrivals_mean": "blocked_arrivals",
    "max_utilization_ratio_mean": "max_utilization",
    "total_overflow_excess_mean": "overflow",
    "num_unsafe_rows_mean": "unsafe_rows",
    "total_surge_gap_mean": "surge_gap",
}


# ============================================================
# Rankings by scenario
# ============================================================

def build_policy_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []
    scenarios = summary_df["scenario"].dropna().unique().tolist()

    for scenario in scenarios:
        sdf = summary_df[summary_df["scenario"] == scenario].copy()

        for metric in MINIMIZE_METRICS:
            if metric not in sdf.columns:
                continue

            tmp = sdf[["policy_name", metric]].copy()
            tmp = tmp.sort_values(metric, ascending=True).reset_index(drop=True)

            for rank_idx, row in tmp.iterrows():
                rows.append(
                    {
                        "scenario": scenario,
                        "metric": PRETTY_NAMES.get(metric, metric),
                        "policy_name": row["policy_name"],
                        "rank": rank_idx + 1,
                        "value": float(row[metric]),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["rank"] = out["rank"].astype(int)
    return out


# ============================================================
# Win counts
# ============================================================

def build_policy_win_counts(rank_df: pd.DataFrame) -> pd.DataFrame:
    if rank_df.empty:
        return pd.DataFrame()

    wins = rank_df[rank_df["rank"] == 1].copy()

    out = (
        wins.groupby("policy_name")
        .size()
        .reset_index(name="num_metric_wins")
        .sort_values(
            ["num_metric_wins", "policy_name"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return out


# ============================================================
# Average rank across scenarios
# ============================================================

def build_average_rank_table(rank_df: pd.DataFrame) -> pd.DataFrame:
    if rank_df.empty:
        return pd.DataFrame()

    out = (
        rank_df.groupby(["policy_name", "metric"])["rank"]
        .mean()
        .reset_index(name="avg_rank")
        .sort_values(
            ["metric", "avg_rank", "policy_name"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )

    return out


# ============================================================
# Tradeoff frontier
# ============================================================

def build_tradeoff_frontier(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    needed = [
        "scenario",
        "policy_name",
        "total_unsafe_excess_mean",
        "total_blocked_arrivals_mean",
        "max_utilization_ratio_mean",
        "total_overflow_excess_mean",
    ]

    cols = [c for c in needed if c in summary_df.columns]
    out = summary_df[cols].copy()

    if (
        "total_unsafe_excess_mean" in out.columns
        and "total_blocked_arrivals_mean" in out.columns
    ):
        out["safety_access_ratio"] = (
            out["total_unsafe_excess_mean"]
            / (1.0 + out["total_blocked_arrivals_mean"])
        )

    return out.sort_values(
        ["scenario", "total_unsafe_excess_mean", "policy_name"]
    ).reset_index(drop=True)


# ============================================================
# Relative improvements
# ============================================================

def one_relative_change(base: float, challenger: float) -> float:
    if pd.isna(base):
        return np.nan
    if abs(base) < 1e-12:
        return np.nan
    return (base - challenger) / base


def build_relative_improvements(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []
    scenarios = summary_df["scenario"].dropna().unique().tolist()

    compare_pairs = [
        ("optimized_network", "regime_robust_optimized_network"),
        ("robust_optimized_network", "regime_robust_optimized_network"),
        ("optimized_network", "robust_optimized_network"),
    ]

    metrics = [
        "total_unsafe_excess_mean",
        "total_blocked_arrivals_mean",
        "max_utilization_ratio_mean",
        "total_overflow_excess_mean",
    ]

    for scenario in scenarios:
        sdf = summary_df[summary_df["scenario"] == scenario].copy()

        for base_policy, challenger_policy in compare_pairs:
            bdf = sdf[sdf["policy_name"] == base_policy]
            cdf = sdf[sdf["policy_name"] == challenger_policy]

            if bdf.empty or cdf.empty:
                continue

            b = bdf.iloc[0]
            c = cdf.iloc[0]

            row = {
                "scenario": scenario,
                "base_policy": base_policy,
                "challenger_policy": challenger_policy,
            }

            for metric in metrics:
                if metric not in sdf.columns:
                    continue

                pretty = PRETTY_NAMES.get(metric, metric)
                row[f"{pretty}_relative_improvement"] = one_relative_change(
                    float(b[metric]),
                    float(c[metric]),
                )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Executive summary
# ============================================================

def build_executive_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []
    scenarios = summary_df["scenario"].dropna().unique().tolist()

    for scenario in scenarios:
        sdf = summary_df[summary_df["scenario"] == scenario].copy()

        if sdf.empty:
            continue

        best_safety = sdf.sort_values("total_unsafe_excess_mean").iloc[0]
        best_access = sdf.sort_values("total_blocked_arrivals_mean").iloc[0]

        tmp = sdf.copy()
        tmp["unsafe_z"] = (
            tmp["total_unsafe_excess_mean"]
            - tmp["total_unsafe_excess_mean"].mean()
        ) / (tmp["total_unsafe_excess_mean"].std(ddof=0) + 1e-9)

        tmp["blocked_z"] = (
            tmp["total_blocked_arrivals_mean"]
            - tmp["total_blocked_arrivals_mean"].mean()
        ) / (tmp["total_blocked_arrivals_mean"].std(ddof=0) + 1e-9)

        tmp["balanced_score"] = tmp["unsafe_z"] + tmp["blocked_z"]
        best_balanced = tmp.sort_values("balanced_score").iloc[0]

        rows.append(
            {
                "scenario": scenario,
                "best_safety_policy": best_safety["policy_name"],
                "best_safety_value": float(best_safety["total_unsafe_excess_mean"]),
                "best_access_policy": best_access["policy_name"],
                "best_access_value": float(best_access["total_blocked_arrivals_mean"]),
                "recommended_compromise_policy": best_balanced["policy_name"],
                "balanced_score": float(best_balanced["balanced_score"]),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Regime-specific headline summary
# ============================================================

def build_regime_headline_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    regime_rows = summary_df[
        summary_df["policy_name"] == "regime_robust_optimized_network"
    ].copy()

    nominal_rows = summary_df[
        summary_df["policy_name"] == "optimized_network"
    ].copy()

    robust_rows = summary_df[
        summary_df["policy_name"] == "robust_optimized_network"
    ].copy()

    rows = []

    for _, rr in regime_rows.iterrows():
        scenario = rr["scenario"]

        nr = nominal_rows[nominal_rows["scenario"] == scenario]
        br = robust_rows[robust_rows["scenario"] == scenario]

        if nr.empty:
            continue

        nr = nr.iloc[0]
        br = br.iloc[0] if not br.empty else None

        row = {
            "scenario": scenario,
            "regime_unsafe": float(rr["total_unsafe_excess_mean"]),
            "nominal_unsafe": float(nr["total_unsafe_excess_mean"]),
            "regime_vs_nominal_gain": float(nr["total_unsafe_excess_mean"]) - float(rr["total_unsafe_excess_mean"]),
        }

        if br is not None:
            row["robust_unsafe"] = float(br["total_unsafe_excess_mean"])
            row["regime_vs_robust_gain"] = float(br["total_unsafe_excess_mean"]) - float(rr["total_unsafe_excess_mean"])

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    summary_df = safe_read_csv(SUMMARY_PATH)
    _ = safe_read_csv(REPL_PATH)
    _ = safe_read_csv(DELTA_PATH)

    if summary_df.empty:
        print("Missing regime_suite_summary.csv")
        return

    rankings_df = build_policy_rankings(summary_df)
    wins_df = build_policy_win_counts(rankings_df)
    avg_rank_df = build_average_rank_table(rankings_df)
    frontier_df = build_tradeoff_frontier(summary_df)
    rel_imp_df = build_relative_improvements(summary_df)
    exec_df = build_executive_summary(summary_df)
    headline_df = build_regime_headline_table(summary_df)

    out1 = TABLES_DIR / "regime_policy_rankings.csv"
    out2 = TABLES_DIR / "regime_policy_win_counts.csv"
    out3 = TABLES_DIR / "regime_policy_average_ranks.csv"
    out4 = TABLES_DIR / "regime_tradeoff_frontier.csv"
    out5 = TABLES_DIR / "regime_relative_improvements.csv"
    out6 = TABLES_DIR / "regime_executive_summary.csv"
    out7 = TABLES_DIR / "regime_headline_summary.csv"

    rankings_df.to_csv(out1, index=False)
    wins_df.to_csv(out2, index=False)
    avg_rank_df.to_csv(out3, index=False)
    frontier_df.to_csv(out4, index=False)
    rel_imp_df.to_csv(out5, index=False)
    exec_df.to_csv(out6, index=False)
    headline_df.to_csv(out7, index=False)

    print("\nSaved comparison tables:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)
    print(out5)
    print(out6)
    print(out7)

    print("\n=== Policy Win Counts ===")
    if not wins_df.empty:
        print(wins_df.to_string(index=False))

    print("\n=== Executive Summary ===")
    if not exec_df.empty:
        print(exec_df.to_string(index=False))

    print("\n=== Regime Headline Summary ===")
    if not headline_df.empty:
        print(headline_df.to_string(index=False))


if __name__ == "__main__":
    main()