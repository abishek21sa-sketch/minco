from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCENARIO_ORDER = [
    "baseline",
    "h3_c4_shock_1p5",
    "h3_c4_shock_2p0",
    "h3_icu_capacity_reduced",
    "transfer_disabled",
]

POLICY_ORDER = [
    "local_only",
    "no_control",
    "no_transfer",
    "optimized_network",
    "myopic_milp"
]


def _load_scenario_suite_summary(
    csv_path: str | Path = "results/scenario_suite_summary.csv",
) -> pd.DataFrame:
    """
    Load and order the scenario-suite summary table.
    """
    df = pd.read_csv(csv_path).copy()

    df["scenario"] = pd.Categorical(
        df["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True,
    )
    df["policy_name"] = pd.Categorical(
        df["policy_name"],
        categories=POLICY_ORDER,
        ordered=True,
    )

    df = df.sort_values(["scenario", "policy_name"]).reset_index(drop=True)
    return df


def _plot_metric_lines(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one line per policy across scenarios.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    scenarios = [s for s in SCENARIO_ORDER if s in set(df["scenario"].astype(str))]

    for policy in POLICY_ORDER:
        sub = df[df["policy_name"].astype(str) == policy].copy()
        if sub.empty:
            continue

        sub = (
            sub.set_index("scenario")
            .reindex(scenarios)
            .reset_index()
        )

        ax.plot(
            range(len(scenarios)),
            sub[metric_col],
            marker="o",
            label=policy,
        )

    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=20)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bar(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot grouped bars by scenario, with one bar per policy.
    """
    fig, ax = plt.subplots(figsize=(11, 5.8))

    scenarios = [s for s in SCENARIO_ORDER if s in set(df["scenario"].astype(str))]
    policies = [p for p in POLICY_ORDER if p in set(df["policy_name"].astype(str))]

    x = list(range(len(scenarios)))
    width = 0.18

    for idx, policy in enumerate(policies):
        sub = df[df["policy_name"].astype(str) == policy].copy()
        sub = (
            sub.set_index("scenario")
            .reindex(scenarios)
            .reset_index()
        )

        offsets = [i + (idx - (len(policies) - 1) / 2) * width for i in x]

        ax.bar(
            offsets,
            sub[metric_col],
            width=width,
            label=policy,
        )

    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_unsafe_excess_lines(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Line plot of mean total unsafe excess across scenarios.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_suite_total_unsafe_excess_lines.png"

    _plot_metric_lines(
        df=df,
        metric_col="total_unsafe_excess_mean",
        title="Mean Total Unsafe Excess Across Scenarios",
        ylabel="Mean total unsafe excess",
        output_path=output_path,
    )
    return output_path


def plot_max_utilization_lines(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Line plot of mean max utilization ratio across scenarios.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_suite_max_utilization_lines.png"

    _plot_metric_lines(
        df=df,
        metric_col="max_utilization_ratio_mean",
        title="Mean Max Utilization Ratio Across Scenarios",
        ylabel="Mean max utilization ratio",
        output_path=output_path,
    )
    return output_path


def plot_blocked_arrivals_lines(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Line plot of mean total blocked arrivals across scenarios.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_suite_blocked_arrivals_lines.png"

    _plot_metric_lines(
        df=df,
        metric_col="total_blocked_arrivals_mean",
        title="Mean Blocked Arrivals Across Scenarios",
        ylabel="Mean total blocked arrivals",
        output_path=output_path,
    )
    return output_path


def plot_unsafe_excess_bars(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Grouped bar chart of mean total unsafe excess.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_suite_total_unsafe_excess_bars.png"

    _plot_grouped_bar(
        df=df,
        metric_col="total_unsafe_excess_mean",
        title="Mean Total Unsafe Excess by Scenario and Policy",
        ylabel="Mean total unsafe excess",
        output_path=output_path,
    )
    return output_path


def plot_max_utilization_bars(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Grouped bar chart of mean max utilization ratio.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_suite_max_utilization_bars.png"

    _plot_grouped_bar(
        df=df,
        metric_col="max_utilization_ratio_mean",
        title="Mean Max Utilization Ratio by Scenario and Policy",
        ylabel="Mean max utilization ratio",
        output_path=output_path,
    )
    return output_path


def plot_num_unsafe_rows_bars(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Grouped bar chart of mean unsafe rows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scenario_suite_num_unsafe_rows_bars.png"

    _plot_grouped_bar(
        df=df,
        metric_col="num_unsafe_rows_mean",
        title="Mean Unsafe Capacity Events by Scenario and Policy",
        ylabel="Mean number of unsafe rows",
        output_path=output_path,
    )
    return output_path


def generate_all_scenario_suite_plots(
    csv_path: str | Path = "results/scenario_suite_summary.csv",
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """
    Generate all standard scenario-suite plots.
    """
    df = _load_scenario_suite_summary(csv_path)

    outputs = [
        plot_unsafe_excess_lines(df, output_dir=output_dir),
        plot_max_utilization_lines(df, output_dir=output_dir),
        plot_blocked_arrivals_lines(df, output_dir=output_dir),
        plot_unsafe_excess_bars(df, output_dir=output_dir),
        plot_max_utilization_bars(df, output_dir=output_dir),
        plot_num_unsafe_rows_bars(df, output_dir=output_dir),
    ]
    return outputs


def main() -> None:
    outputs = generate_all_scenario_suite_plots()

    print("\nGenerated scenario suite figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()