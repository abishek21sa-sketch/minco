from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["synthetic", "literature_calibrated"]
POLICY_ORDER = [
    "optimized_network",
    "myopic_milp",
    "no_transfer",
    "local_only",
    "no_control",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_inputs() -> dict[str, dict[str, pd.DataFrame]]:
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for mode in MODES:
        data[mode] = {
            "scenario_summary": safe_read_csv(RESULTS_DIR / f"scenario_suite_summary_{mode}.csv"),
            "hospital_bottleneck_summary": safe_read_csv(
                RESULTS_DIR / f"hospital_bottleneck_summary_{mode}.csv"
            ),
            "hospital_time_series_outputs": safe_read_csv(
                RESULTS_DIR / f"hospital_time_series_outputs_{mode}.csv"
            ),
            "statistical_tests_executive": safe_read_csv(
                TABLES_DIR / f"statistical_tests_executive_{mode}.csv"
            ),
        }
    return data


def _ordered_policies(df: pd.DataFrame, policy_col: str = "policy_name") -> list[str]:
    present = df[policy_col].dropna().unique().tolist()
    ordered = [p for p in POLICY_ORDER if p in present]
    extras = sorted([p for p in present if p not in ordered])
    return ordered + extras


def plot_policy_metric_bars(
    df: pd.DataFrame,
    scenario: str,
    mode: str,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    filename: str,
) -> None:
    sdf = df[df["scenario"] == scenario].copy()
    if sdf.empty:
        print(f"Skipping {filename}: no rows for scenario={scenario}, mode={mode}")
        return

    order = _ordered_policies(sdf)
    sdf["policy_name"] = pd.Categorical(sdf["policy_name"], categories=order, ordered=True)
    sdf = sdf.sort_values("policy_name")

    x = range(len(sdf))
    y = sdf[metric_mean_col].astype(float).tolist()
    e = sdf[metric_std_col].astype(float).tolist()

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, yerr=e, capsize=5)
    plt.xticks(list(x), sdf["policy_name"].astype(str).tolist(), rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("Policy")
    plt.title(f"{ylabel} by Policy | {scenario} | {mode}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_blocked_arrivals_mode_comparison(
    all_data: dict[str, dict[str, pd.DataFrame]],
    scenario: str,
    filename: str,
) -> None:
    rows = []
    for mode, data in all_data.items():
        df = data["scenario_summary"]
        sdf = df[df["scenario"] == scenario].copy()
        if sdf.empty:
            continue
        sdf["mode"] = mode
        rows.append(sdf)

    if not rows:
        print(f"Skipping {filename}: no data for scenario={scenario}")
        return

    stacked = pd.concat(rows, ignore_index=True)
    order = _ordered_policies(stacked)

    plt.figure(figsize=(11, 6))

    width = 0.38
    x = list(range(len(order)))

    for j, mode in enumerate(MODES):
        mdf = stacked[stacked["mode"] == mode].copy()
        mdf["policy_name"] = pd.Categorical(mdf["policy_name"], categories=order, ordered=True)
        mdf = mdf.sort_values("policy_name")

        xpos = [i + (j - 0.5) * width for i in x]
        y = mdf["total_blocked_arrivals_mean"].astype(float).tolist()
        e = mdf["total_blocked_arrivals_std"].astype(float).tolist()

        plt.bar(
            xpos,
            y,
            width=width,
            yerr=e,
            capsize=4,
            label=mode,
        )

    plt.xticks(x, order, rotation=25, ha="right")
    plt.ylabel("Mean blocked arrivals")
    plt.xlabel("Policy")
    plt.title(f"Blocked Arrivals by Policy | {scenario} | Synthetic vs Literature-Calibrated")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hospital_bottleneck_heatmap(
    df: pd.DataFrame,
    scenario: str,
    mode: str,
    filename: str,
) -> None:
    sdf = df[df["scenario"] == scenario].copy()
    if sdf.empty:
        print(f"Skipping {filename}: no rows for scenario={scenario}, mode={mode}")
        return

    pivot = sdf.pivot(
        index="hospital_id",
        columns="policy_name",
        values="bottleneck_score_mean",
    )

    order = [p for p in POLICY_ORDER if p in pivot.columns] + [
        p for p in pivot.columns if p not in POLICY_ORDER
    ]
    pivot = pivot.reindex(columns=order)

    plt.figure(figsize=(10, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Mean bottleneck score")
    plt.xticks(range(len(pivot.columns)), pivot.columns.tolist(), rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.tolist())
    plt.xlabel("Policy")
    plt.ylabel("Hospital")
    plt.title(f"Hospital Bottleneck Heatmap | {scenario} | {mode}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


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

    out = (
        sdf.groupby(["policy_name", "day"], as_index=False)
        .agg(
            mean_bottleneck_score=("bottleneck_score", "mean"),
            std_bottleneck_score=("bottleneck_score", "std"),
        )
    )
    out["std_bottleneck_score"] = out["std_bottleneck_score"].fillna(0.0)
    return out.sort_values(["policy_name", "day"]).reset_index(drop=True)


def plot_temporal_congestion_curves(
    df: pd.DataFrame,
    scenario: str,
    hospital_id: str,
    mode: str,
    filename: str,
) -> None:
    agg = aggregate_temporal(df, scenario=scenario, hospital_id=hospital_id)
    if agg.empty:
        print(f"Skipping {filename}: no temporal rows for scenario={scenario}, hospital={hospital_id}, mode={mode}")
        return

    order = _ordered_policies(agg, policy_col="policy_name")

    plt.figure(figsize=(11, 6))

    for policy in order:
        pdf = agg[agg["policy_name"] == policy].sort_values("day")
        if pdf.empty:
            continue

        x = pdf["day"].to_numpy()
        y = pdf["mean_bottleneck_score"].to_numpy()
        s = pdf["std_bottleneck_score"].to_numpy()
        n = max(len(pdf), 1)
        half = 1.96 * s / (n ** 0.5)

        plt.plot(x, y, marker="o", linewidth=2.2, label=policy)
        plt.fill_between(x, y - half, y + half, alpha=0.18)

    plt.xlabel("Day")
    plt.ylabel("Mean bottleneck score")
    plt.title(f"Temporal Congestion Curves | {scenario} | {hospital_id} | {mode}")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_effect_size_forest(
    df: pd.DataFrame,
    scenario: str,
    mode: str,
    metric: str,
    filename: str,
) -> None:
    sdf = df[
        (df["scenario"] == scenario)
        & (df["metric"] == metric)
    ].copy()

    if sdf.empty:
        print(f"Skipping {filename}: no statistical rows for scenario={scenario}, mode={mode}, metric={metric}")
        return

    sdf = sdf.sort_values("cohens_d")
    labels = [f"{a} vs {b}" for a, b in zip(sdf["policy_a"], sdf["policy_b"])]
    x = sdf["cohens_d"].astype(float).tolist()
    y = list(range(len(labels)))

    plt.figure(figsize=(10, 6))
    plt.axvline(0.0, linewidth=1.2)
    plt.scatter(x, y)
    plt.yticks(y, labels)
    plt.xlabel("Cohen's d")
    plt.title(f"Effect Size Forest Plot | {metric} | {scenario} | {mode}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    all_data = load_inputs()

    # 1. Policy KPI bar charts
    for mode, data in all_data.items():
        scenario_summary = data["scenario_summary"]

        for scenario in sorted(scenario_summary["scenario"].dropna().unique().tolist()):
            plot_policy_metric_bars(
                df=scenario_summary,
                scenario=scenario,
                mode=mode,
                metric_mean_col="total_unsafe_excess_mean",
                metric_std_col="total_unsafe_excess_std",
                ylabel="Mean unsafe excess",
                filename=f"pub_unsafe_excess_bars_{scenario}_{mode}.png",
            )

            plot_policy_metric_bars(
                df=scenario_summary,
                scenario=scenario,
                mode=mode,
                metric_mean_col="max_utilization_ratio_mean",
                metric_std_col="max_utilization_ratio_std",
                ylabel="Mean max utilization ratio",
                filename=f"pub_max_utilization_bars_{scenario}_{mode}.png",
            )

    # 2. Blocked arrivals comparison by mode
    for scenario in ["baseline", "h3_icu_capacity_reduced", "transfer_disabled"]:
        plot_blocked_arrivals_mode_comparison(
            all_data=all_data,
            scenario=scenario,
            filename=f"pub_blocked_arrivals_compare_modes_{scenario}.png",
        )

    # 3. Bottleneck heatmaps
    for mode, data in all_data.items():
        bottleneck_df = data["hospital_bottleneck_summary"]
        for scenario in sorted(bottleneck_df["scenario"].dropna().unique().tolist()):
            plot_hospital_bottleneck_heatmap(
                df=bottleneck_df,
                scenario=scenario,
                mode=mode,
                filename=f"pub_bottleneck_heatmap_{scenario}_{mode}.png",
            )

    # 4. Temporal congestion curves for H3
    for mode, data in all_data.items():
        ts_df = data["hospital_time_series_outputs"]
        for scenario in sorted(ts_df["scenario"].dropna().unique().tolist()):
            plot_temporal_congestion_curves(
                df=ts_df,
                scenario=scenario,
                hospital_id="H3",
                mode=mode,
                filename=f"pub_temporal_h3_{scenario}_{mode}.png",
            )

    # 5. Effect size forest plots
    for mode, data in all_data.items():
        stat_df = data["statistical_tests_executive"]
        for scenario in sorted(stat_df["scenario"].dropna().unique().tolist()):
            for metric in ["total_unsafe_excess", "max_utilization_ratio", "total_blocked_arrivals"]:
                plot_effect_size_forest(
                    df=stat_df,
                    scenario=scenario,
                    mode=mode,
                    metric=metric,
                    filename=f"pub_forest_{metric}_{scenario}_{mode}.png",
                )

    print("\nSaved publication figures to:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()