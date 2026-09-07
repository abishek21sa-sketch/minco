from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RESULTS_DIR / "hospital_time_series_outputs.csv"


def load_temporal_outputs() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}. Run the scenario suite first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_cols = {
        "scenario",
        "policy_name",
        "replication",
        "day",
        "hospital_id",
        "bottleneck_score",
        "unsafe_excess",
        "overflow_excess",
        "max_utilization",
        "transfer_activity",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"hospital_time_series_outputs.csv missing columns: {sorted(missing)}"
        )

    return df


def _add_ci_columns(
    agg: pd.DataFrame,
    mean_col: str,
    std_col: str,
    n_col: str,
    ci_prefix: str,
) -> pd.DataFrame:
    out = agg.copy()
    se = out[std_col] / out[n_col].pow(0.5)
    ci_half = 1.96 * se
    out[f"{ci_prefix}_lower"] = out[mean_col] - ci_half
    out[f"{ci_prefix}_upper"] = out[mean_col] + ci_half
    return out


def aggregate_temporal(
    df: pd.DataFrame,
    scenario: str,
    hospital_id: str,
) -> pd.DataFrame:
    sdf = df[
        (df["scenario"] == scenario)
        & (df["hospital_id"] == hospital_id)
    ].copy()

    if sdf.empty:
        return pd.DataFrame()

    agg = (
        sdf.groupby(["scenario", "policy_name", "hospital_id", "day"], as_index=False)
        .agg(
            n_replications=("replication", "nunique"),
            mean_bottleneck_score=("bottleneck_score", "mean"),
            std_bottleneck_score=("bottleneck_score", "std"),
            mean_unsafe_excess=("unsafe_excess", "mean"),
            std_unsafe_excess=("unsafe_excess", "std"),
            mean_overflow_excess=("overflow_excess", "mean"),
            std_overflow_excess=("overflow_excess", "std"),
            mean_utilization=("max_utilization", "mean"),
            std_utilization=("max_utilization", "std"),
            mean_transfer_activity=("transfer_activity", "mean"),
            std_transfer_activity=("transfer_activity", "std"),
            p95_bottleneck_score=("bottleneck_score", lambda s: float(s.quantile(0.95))),
            p95_unsafe_excess=("unsafe_excess", lambda s: float(s.quantile(0.95))),
            p95_utilization=("max_utilization", lambda s: float(s.quantile(0.95))),
        )
    )

    std_cols = [c for c in agg.columns if c.startswith("std_")]
    agg[std_cols] = agg[std_cols].fillna(0.0)

    agg = _add_ci_columns(
        agg, "mean_bottleneck_score", "std_bottleneck_score", "n_replications", "bottleneck_ci"
    )
    agg = _add_ci_columns(
        agg, "mean_unsafe_excess", "std_unsafe_excess", "n_replications", "unsafe_ci"
    )
    agg = _add_ci_columns(
        agg, "mean_utilization", "std_utilization", "n_replications", "util_ci"
    )
    agg = _add_ci_columns(
        agg, "mean_overflow_excess", "std_overflow_excess", "n_replications", "overflow_ci"
    )
    agg = _add_ci_columns(
        agg, "mean_transfer_activity", "std_transfer_activity", "n_replications", "transfer_ci"
    )

    return agg.sort_values(["policy_name", "day"]).reset_index(drop=True)


def aggregate_temporal_multi_scenario(
    df: pd.DataFrame,
    scenarios: list[str],
    policy_name: str,
    hospital_id: str,
) -> pd.DataFrame:
    sdf = df[
        (df["scenario"].isin(scenarios))
        & (df["policy_name"] == policy_name)
        & (df["hospital_id"] == hospital_id)
    ].copy()

    if sdf.empty:
        return pd.DataFrame()

    agg = (
        sdf.groupby(["scenario", "policy_name", "hospital_id", "day"], as_index=False)
        .agg(
            n_replications=("replication", "nunique"),
            mean_bottleneck_score=("bottleneck_score", "mean"),
            std_bottleneck_score=("bottleneck_score", "std"),
            mean_unsafe_excess=("unsafe_excess", "mean"),
            std_unsafe_excess=("unsafe_excess", "std"),
            mean_utilization=("max_utilization", "mean"),
            std_utilization=("max_utilization", "std"),
            mean_transfer_activity=("transfer_activity", "mean"),
            std_transfer_activity=("transfer_activity", "std"),
        )
    )

    std_cols = [c for c in agg.columns if c.startswith("std_")]
    agg[std_cols] = agg[std_cols].fillna(0.0)

    agg = _add_ci_columns(
        agg, "mean_bottleneck_score", "std_bottleneck_score", "n_replications", "bottleneck_ci"
    )
    agg = _add_ci_columns(
        agg, "mean_unsafe_excess", "std_unsafe_excess", "n_replications", "unsafe_ci"
    )
    agg = _add_ci_columns(
        agg, "mean_utilization", "std_utilization", "n_replications", "util_ci"
    )
    agg = _add_ci_columns(
        agg, "mean_transfer_activity", "std_transfer_activity", "n_replications", "transfer_ci"
    )

    return agg.sort_values(["scenario", "day"]).reset_index(drop=True)


def build_peak_metrics_table(
    df: pd.DataFrame,
    hospital_id: str,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    working = df[df["hospital_id"] == hospital_id].copy()

    if scenarios is not None:
        working = working[working["scenario"].isin(scenarios)].copy()

    if working.empty:
        return pd.DataFrame()

    daily = (
        working.groupby(["scenario", "policy_name", "hospital_id", "day"], as_index=False)
        .agg(
            mean_bottleneck_score=("bottleneck_score", "mean"),
            mean_unsafe_excess=("unsafe_excess", "mean"),
            mean_utilization=("max_utilization", "mean"),
        )
    )

    peak_idx = (
        daily.groupby(["scenario", "policy_name", "hospital_id"])["mean_bottleneck_score"]
        .idxmax()
    )

    peak_rows = (
        daily.loc[
            peak_idx,
            [
                "scenario",
                "policy_name",
                "hospital_id",
                "day",
                "mean_bottleneck_score",
                "mean_unsafe_excess",
                "mean_utilization",
            ],
        ]
        .rename(
            columns={
                "day": "peak_day",
                "mean_bottleneck_score": "peak_mean_bottleneck",
                "mean_unsafe_excess": "unsafe_excess_at_peak",
                "mean_utilization": "utilization_at_peak",
            }
        )
        .reset_index(drop=True)
    )

    final_day = int(daily["day"].max())
    final_rows = (
        daily[daily["day"] == final_day][
            [
                "scenario",
                "policy_name",
                "hospital_id",
                "mean_bottleneck_score",
                "mean_unsafe_excess",
                "mean_utilization",
            ]
        ]
        .rename(
            columns={
                "mean_bottleneck_score": "final_day_bottleneck",
                "mean_unsafe_excess": "final_day_unsafe_excess",
                "mean_utilization": "final_day_utilization",
            }
        )
        .reset_index(drop=True)
    )

    averages = (
        daily.groupby(["scenario", "policy_name", "hospital_id"], as_index=False)
        .agg(
            mean_daily_bottleneck=("mean_bottleneck_score", "mean"),
            mean_daily_unsafe_excess=("mean_unsafe_excess", "mean"),
            mean_daily_utilization=("mean_utilization", "mean"),
        )
    )

    out = averages.merge(
        peak_rows,
        on=["scenario", "policy_name", "hospital_id"],
        how="left",
    ).merge(
        final_rows,
        on=["scenario", "policy_name", "hospital_id"],
        how="left",
    )

    return out.sort_values(
        ["scenario", "peak_mean_bottleneck"],
        ascending=[True, False],
    ).reset_index(drop=True)


def plot_metric_by_policy(
    agg_df: pd.DataFrame,
    mean_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    policy_order: list[str] | None = None,
) -> None:
    if agg_df.empty:
        print(f"Skipping {output_path.name}: no data.")
        return

    plt.figure(figsize=(11, 6))

    if policy_order is None:
        policy_order = sorted(agg_df["policy_name"].unique().tolist())

    for policy in policy_order:
        pdf = agg_df[agg_df["policy_name"] == policy].sort_values("day")
        if pdf.empty:
            continue

        x = pdf["day"].to_numpy()
        y = pdf[mean_col].to_numpy()
        lo = pdf[ci_lower_col].to_numpy()
        hi = pdf[ci_upper_col].to_numpy()

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            label=policy,
        )
        plt.fill_between(
            x,
            lo,
            hi,
            alpha=0.20,
        )

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel(ylabel)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_by_scenario(
    agg_df: pd.DataFrame,
    mean_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    scenario_order: list[str] | None = None,
) -> None:
    if agg_df.empty:
        print(f"Skipping {output_path.name}: no data.")
        return

    plt.figure(figsize=(11, 6))

    if scenario_order is None:
        scenario_order = sorted(agg_df["scenario"].unique().tolist())

    for scenario in scenario_order:
        sdf = agg_df[agg_df["scenario"] == scenario].sort_values("day")
        if sdf.empty:
            continue

        x = sdf["day"].to_numpy()
        y = sdf[mean_col].to_numpy()
        lo = sdf[ci_lower_col].to_numpy()
        hi = sdf[ci_upper_col].to_numpy()

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            label=scenario,
        )
        plt.fill_between(
            x,
            lo,
            hi,
            alpha=0.20,
        )

    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel(ylabel)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_temporal_outputs()

    hospital_id = "H3"
    policy_order = [
        "optimized_network",
        "myopic_milp",
        "no_transfer",
        "local_only",
        "no_control",
    ]

    peak_metrics = build_peak_metrics_table(df=df, hospital_id=hospital_id)
    peak_metrics_path = TABLES_DIR / "temporal_peak_metrics.csv"
    peak_metrics.to_csv(peak_metrics_path, index=False)

    baseline_agg = aggregate_temporal(
        df=df,
        scenario="baseline",
        hospital_id=hospital_id,
    )

    plot_metric_by_policy(
        agg_df=baseline_agg,
        mean_col="mean_bottleneck_score",
        ci_lower_col="bottleneck_ci_lower",
        ci_upper_col="bottleneck_ci_upper",
        title="H3 Mean Bottleneck Score by Day Across Policies (Baseline)",
        ylabel="Mean bottleneck score",
        output_path=FIGURES_DIR / "paper_h3_baseline_bottleneck_by_policy.png",
        policy_order=policy_order,
    )

    plot_metric_by_policy(
        agg_df=baseline_agg,
        mean_col="mean_unsafe_excess",
        ci_lower_col="unsafe_ci_lower",
        ci_upper_col="unsafe_ci_upper",
        title="H3 Mean Unsafe Excess by Day Across Policies (Baseline)",
        ylabel="Mean unsafe excess",
        output_path=FIGURES_DIR / "paper_h3_baseline_unsafe_excess_by_policy.png",
        policy_order=policy_order,
    )

    plot_metric_by_policy(
        agg_df=baseline_agg,
        mean_col="mean_utilization",
        ci_lower_col="util_ci_lower",
        ci_upper_col="util_ci_upper",
        title="H3 Mean Utilization by Day Across Policies (Baseline)",
        ylabel="Mean utilization ratio",
        output_path=FIGURES_DIR / "paper_h3_baseline_utilization_by_policy.png",
        policy_order=policy_order,
    )

    optimized_capacity_compare = aggregate_temporal_multi_scenario(
        df=df,
        scenarios=["baseline", "h3_icu_capacity_reduced"],
        policy_name="optimized_network",
        hospital_id=hospital_id,
    )

    plot_metric_by_scenario(
        agg_df=optimized_capacity_compare,
        mean_col="mean_bottleneck_score",
        ci_lower_col="bottleneck_ci_lower",
        ci_upper_col="bottleneck_ci_upper",
        title="H3 Mean Bottleneck Score by Day: Baseline vs H3 ICU Capacity Reduced (Optimized Network)",
        ylabel="Mean bottleneck score",
        output_path=FIGURES_DIR / "paper_h3_optimized_baseline_vs_capacity_reduced_bottleneck.png",
        scenario_order=["baseline", "h3_icu_capacity_reduced"],
    )

    optimized_transfer_compare = aggregate_temporal_multi_scenario(
        df=df,
        scenarios=["baseline", "transfer_disabled"],
        policy_name="optimized_network",
        hospital_id=hospital_id,
    )

    plot_metric_by_scenario(
        agg_df=optimized_transfer_compare,
        mean_col="mean_bottleneck_score",
        ci_lower_col="bottleneck_ci_lower",
        ci_upper_col="bottleneck_ci_upper",
        title="H3 Mean Bottleneck Score by Day: Baseline vs Transfer Disabled (Optimized Network)",
        ylabel="Mean bottleneck score",
        output_path=FIGURES_DIR / "paper_h3_optimized_baseline_vs_transfer_disabled_bottleneck.png",
        scenario_order=["baseline", "transfer_disabled"],
    )

    print("\nGenerated temporal paper figures:")
    for path in sorted(FIGURES_DIR.glob("paper_h3_*.png")):
        print(path)

    print("\nGenerated temporal peak metrics table:")
    print(peak_metrics_path)


if __name__ == "__main__":
    main()