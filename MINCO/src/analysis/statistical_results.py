from __future__ import annotations

from pathlib import Path
from math import erf, sqrt

import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_REPLICATIONS_FILE = RESULTS_DIR / "scenario_suite_replications.csv"
TEMPORAL_OUTPUTS_FILE = RESULTS_DIR / "hospital_time_series_outputs.csv"


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def two_sided_pvalue_from_z(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_df = safe_read_csv(SCENARIO_REPLICATIONS_FILE)
    ts_df = safe_read_csv(TEMPORAL_OUTPUTS_FILE)

    required_rep_cols = {
        "scenario",
        "policy_name",
        "replication",
        "total_unsafe_excess",
        "total_overflow_excess",
        "total_surge_gap",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_blocked_arrivals",
    }
    missing_rep = required_rep_cols - set(rep_df.columns)
    if missing_rep:
        raise ValueError(
            f"scenario_suite_replications.csv missing columns: {sorted(missing_rep)}"
        )

    required_ts_cols = {
        "scenario",
        "policy_name",
        "replication",
        "day",
        "hospital_id",
        "bottleneck_score",
    }
    missing_ts = required_ts_cols - set(ts_df.columns)
    if missing_ts:
        raise ValueError(
            f"hospital_time_series_outputs.csv missing columns: {sorted(missing_ts)}"
        )

    return rep_df, ts_df


def summarize_replication_metrics(
    rep_df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    sdf = rep_df[rep_df["scenario"] == scenario].copy()
    if sdf.empty:
        return pd.DataFrame()

    metrics = [
        "total_unsafe_excess",
        "total_overflow_excess",
        "total_surge_gap",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_blocked_arrivals",
    ]

    grouped = (
        sdf.groupby(["scenario", "policy_name"], as_index=False)
        .agg(
            n_replications=("replication", "nunique"),
            **{
                f"{m}_mean": (m, "mean")
                for m in metrics
            },
            **{
                f"{m}_std": (m, "std")
                for m in metrics
            },
        )
    )

    std_cols = [c for c in grouped.columns if c.endswith("_std")]
    grouped[std_cols] = grouped[std_cols].fillna(0.0)

    return grouped.sort_values("total_unsafe_excess_mean").reset_index(drop=True)


def build_resilience_index(
    ts_df: pd.DataFrame,
    scenario: str,
    hospital_id: str = "H3",
) -> pd.DataFrame:
    sdf = ts_df[
        (ts_df["scenario"] == scenario)
        & (ts_df["hospital_id"] == hospital_id)
    ].copy()

    if sdf.empty:
        return pd.DataFrame()

    # Area under bottleneck curve for each replication
    auc_df = (
        sdf.groupby(["scenario", "policy_name", "replication"], as_index=False)
        .agg(
            resilience_index=("bottleneck_score", "sum"),
            peak_bottleneck=("bottleneck_score", "max"),
        )
    )

    summary = (
        auc_df.groupby(["scenario", "policy_name"], as_index=False)
        .agg(
            n_replications=("replication", "nunique"),
            resilience_index_mean=("resilience_index", "mean"),
            resilience_index_std=("resilience_index", "std"),
            peak_bottleneck_mean=("peak_bottleneck", "mean"),
            peak_bottleneck_std=("peak_bottleneck", "std"),
        )
    )

    summary[["resilience_index_std", "peak_bottleneck_std"]] = summary[
        ["resilience_index_std", "peak_bottleneck_std"]
    ].fillna(0.0)

    return summary.sort_values("resilience_index_mean").reset_index(drop=True)


def compute_improvement_table(
    summary_df: pd.DataFrame,
    reference_policy: str = "no_control",
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    ref = summary_df[summary_df["policy_name"] == reference_policy]
    if ref.empty:
        return pd.DataFrame()

    ref_row = ref.iloc[0]

    out = summary_df.copy()

    improvement_metrics = [
        "total_unsafe_excess_mean",
        "total_overflow_excess_mean",
        "max_utilization_ratio_mean",
        "num_unsafe_rows_mean",
        "total_blocked_arrivals_mean",
    ]

    for col in improvement_metrics:
        ref_val = float(ref_row[col])
        if abs(ref_val) < 1e-12:
            out[f"{col}_pct_improvement_vs_{reference_policy}"] = 0.0
        else:
            out[f"{col}_pct_improvement_vs_{reference_policy}"] = (
                (ref_val - out[col]) / ref_val * 100.0
            )

    return out


def welch_style_policy_test(
    rep_df: pd.DataFrame,
    scenario: str,
    metric: str,
    policy_a: str,
    policy_b: str,
) -> dict:
    sdf = rep_df[rep_df["scenario"] == scenario].copy()
    a = sdf[sdf["policy_name"] == policy_a][metric].dropna().reset_index(drop=True)
    b = sdf[sdf["policy_name"] == policy_b][metric].dropna().reset_index(drop=True)

    if len(a) == 0 or len(b) == 0:
        return {
            "scenario": scenario,
            "metric": metric,
            "policy_a": policy_a,
            "policy_b": policy_b,
            "mean_a": None,
            "mean_b": None,
            "difference_a_minus_b": None,
            "z_like_stat": None,
            "approx_pvalue": None,
        }

    mean_a = float(a.mean())
    mean_b = float(b.mean())
    var_a = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    var_b = float(b.var(ddof=1)) if len(b) > 1 else 0.0

    se = sqrt(var_a / len(a) + var_b / len(b))
    if se < 1e-12:
        z_like = 0.0
        pval = 1.0
    else:
        z_like = (mean_a - mean_b) / se
        pval = two_sided_pvalue_from_z(z_like)

    return {
        "scenario": scenario,
        "metric": metric,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference_a_minus_b": mean_a - mean_b,
        "z_like_stat": z_like,
        "approx_pvalue": pval,
    }


def build_significance_table(
    rep_df: pd.DataFrame,
    scenario: str,
    best_policy: str = "optimized_network",
    comparators: list[str] | None = None,
) -> pd.DataFrame:
    if comparators is None:
        comparators = ["myopic_milp", "no_transfer", "local_only", "no_control"]

    metrics = [
        "total_unsafe_excess",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_blocked_arrivals",
    ]

    rows = []
    for metric in metrics:
        for comp in comparators:
            rows.append(
                welch_style_policy_test(
                    rep_df=rep_df,
                    scenario=scenario,
                    metric=metric,
                    policy_a=best_policy,
                    policy_b=comp,
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["metric", "approx_pvalue"]).reset_index(drop=True)


def build_paper_results_table(
    metric_summary_df: pd.DataFrame,
    resilience_df: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    if metric_summary_df.empty:
        return pd.DataFrame()

    out = metric_summary_df.copy()

    if not resilience_df.empty:
        out = out.merge(
            resilience_df[
                [
                    "scenario",
                    "policy_name",
                    "resilience_index_mean",
                    "resilience_index_std",
                    "peak_bottleneck_mean",
                    "peak_bottleneck_std",
                ]
            ],
            on=["scenario", "policy_name"],
            how="left",
        )

    cols = [
        "scenario",
        "policy_name",
        "n_replications",
        "total_unsafe_excess_mean",
        "total_overflow_excess_mean",
        "max_utilization_ratio_mean",
        "num_unsafe_rows_mean",
        "total_blocked_arrivals_mean",
        "resilience_index_mean",
        "peak_bottleneck_mean",
        f"total_unsafe_excess_mean_pct_improvement_vs_no_control",
        f"max_utilization_ratio_mean_pct_improvement_vs_no_control",
        f"num_unsafe_rows_mean_pct_improvement_vs_no_control",
    ]

    cols = [c for c in cols if c in out.columns]
    out = out[cols].copy()

    return out.sort_values("total_unsafe_excess_mean").reset_index(drop=True)


def main() -> None:
    rep_df, ts_df = load_inputs()

    scenarios = sorted(rep_df["scenario"].dropna().unique().tolist())

    all_summary_tables = []
    all_significance_tables = []
    all_resilience_tables = []

    for scenario in scenarios:
        metric_summary = summarize_replication_metrics(rep_df, scenario)
        resilience_summary = build_resilience_index(ts_df, scenario, hospital_id="H3")
        metric_summary = compute_improvement_table(metric_summary, reference_policy="no_control")
        significance_table = build_significance_table(
            rep_df,
            scenario=scenario,
            best_policy="optimized_network",
            comparators=["myopic_milp", "no_transfer", "local_only", "no_control"],
        )
        paper_table = build_paper_results_table(metric_summary, resilience_summary, scenario)

        metric_path = TABLES_DIR / f"paper_policy_summary_{scenario}.csv"
        sig_path = TABLES_DIR / f"paper_significance_{scenario}.csv"
        resilience_path = TABLES_DIR / f"paper_resilience_{scenario}.csv"

        paper_table.to_csv(metric_path, index=False)
        significance_table.to_csv(sig_path, index=False)
        resilience_summary.to_csv(resilience_path, index=False)

        all_summary_tables.append(paper_table)
        all_significance_tables.append(significance_table)
        all_resilience_tables.append(resilience_summary)

        print(f"\n=== SCENARIO: {scenario} ===")
        print("\nPolicy summary:")
        print(paper_table)
        print("\nSignificance tests:")
        print(significance_table)
        print("\nResilience summary:")
        print(resilience_summary)

    if all_summary_tables:
        combined_summary = pd.concat(all_summary_tables, ignore_index=True)
        combined_summary.to_csv(TABLES_DIR / "paper_policy_summary_all.csv", index=False)

    if all_significance_tables:
        combined_significance = pd.concat(all_significance_tables, ignore_index=True)
        combined_significance.to_csv(TABLES_DIR / "paper_significance_all.csv", index=False)

    if all_resilience_tables:
        combined_resilience = pd.concat(all_resilience_tables, ignore_index=True)
        combined_resilience.to_csv(TABLES_DIR / "paper_resilience_all.csv", index=False)

    print("\nSaved outputs:")
    print(TABLES_DIR / "paper_policy_summary_all.csv")
    print(TABLES_DIR / "paper_significance_all.csv")
    print(TABLES_DIR / "paper_resilience_all.csv")


if __name__ == "__main__":
    main()