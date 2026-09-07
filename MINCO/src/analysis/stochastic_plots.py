from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_validation_tables(
    baseline_csv_path: str | Path = "results/stochastic_validation_baseline_replications.csv",
    shock_csv_path: str | Path = "results/stochastic_validation_h3_c4_shock_replications.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load baseline and shock replication tables and add scenario labels.
    """
    baseline_df = pd.read_csv(baseline_csv_path).copy()
    shock_df = pd.read_csv(shock_csv_path).copy()

    baseline_df["scenario"] = "baseline"
    shock_df["scenario"] = "h3_c4_shock"

    return baseline_df, shock_df


def _combine_for_plotting(
    baseline_df: pd.DataFrame,
    shock_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine baseline and shock tables into one DataFrame.
    """
    combined = pd.concat([baseline_df, shock_df], ignore_index=True)
    combined = combined.reset_index(drop=True)
    return combined


def _plot_boxplot(
    combined_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Create a two-group boxplot for a selected metric.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    baseline_vals = combined_df.loc[
        combined_df["scenario"] == "baseline",
        metric_col,
    ].to_numpy()

    shock_vals = combined_df.loc[
        combined_df["scenario"] == "h3_c4_shock",
        metric_col,
    ].to_numpy()

    ax.boxplot(
        [baseline_vals, shock_vals],
        labels=["Baseline", "H3 c4 Shock"],
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram_overlay(
    combined_df: pd.DataFrame,
    metric_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 12,
) -> None:
    """
    Create an overlaid histogram comparing baseline and shock distributions.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    baseline_vals = combined_df.loc[
        combined_df["scenario"] == "baseline",
        metric_col,
    ].to_numpy()

    shock_vals = combined_df.loc[
        combined_df["scenario"] == "h3_c4_shock",
        metric_col,
    ].to_numpy()

    ax.hist(
        baseline_vals,
        bins=bins,
        alpha=0.6,
        label="Baseline",
    )

    ax.hist(
        shock_vals,
        bins=bins,
        alpha=0.6,
        label="H3 c4 Shock",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_total_unsafe_excess_boxplot(
    combined_df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot boxplot of total unsafe excess.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "boxplot_total_unsafe_excess.png"

    _plot_boxplot(
        combined_df=combined_df,
        metric_col="total_unsafe_excess",
        title="Total Unsafe Excess Across Replications",
        ylabel="Total unsafe excess",
        output_path=output_path,
    )
    return output_path


def plot_max_utilization_ratio_boxplot(
    combined_df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot boxplot of max utilization ratio.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "boxplot_max_utilization_ratio.png"

    _plot_boxplot(
        combined_df=combined_df,
        metric_col="max_utilization_ratio",
        title="Maximum Utilization Ratio Across Replications",
        ylabel="Max utilization ratio",
        output_path=output_path,
    )
    return output_path


def plot_num_unsafe_rows_boxplot(
    combined_df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot boxplot of number of unsafe rows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "boxplot_num_unsafe_rows.png"

    _plot_boxplot(
        combined_df=combined_df,
        metric_col="num_unsafe_rows",
        title="Unsafe Capacity Events Across Replications",
        ylabel="Number of unsafe rows",
        output_path=output_path,
    )
    return output_path


def plot_total_unsafe_excess_histogram(
    combined_df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot histogram overlay of total unsafe excess.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "hist_total_unsafe_excess.png"

    _plot_histogram_overlay(
        combined_df=combined_df,
        metric_col="total_unsafe_excess",
        title="Distribution of Total Unsafe Excess",
        xlabel="Total unsafe excess",
        output_path=output_path,
    )
    return output_path


def plot_max_utilization_ratio_histogram(
    combined_df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot histogram overlay of max utilization ratio.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "hist_max_utilization_ratio.png"

    _plot_histogram_overlay(
        combined_df=combined_df,
        metric_col="max_utilization_ratio",
        title="Distribution of Maximum Utilization Ratio",
        xlabel="Max utilization ratio",
        output_path=output_path,
    )
    return output_path


def generate_all_stochastic_plots(
    baseline_csv_path: str | Path = "results/stochastic_validation_baseline_replications.csv",
    shock_csv_path: str | Path = "results/stochastic_validation_h3_c4_shock_replications.csv",
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """
    Generate all standard stochastic-validation comparison plots.
    """
    baseline_df, shock_df = _load_validation_tables(
        baseline_csv_path=baseline_csv_path,
        shock_csv_path=shock_csv_path,
    )
    combined_df = _combine_for_plotting(
        baseline_df=baseline_df,
        shock_df=shock_df,
    )

    outputs = [
        plot_total_unsafe_excess_boxplot(combined_df, output_dir=output_dir),
        plot_max_utilization_ratio_boxplot(combined_df, output_dir=output_dir),
        plot_num_unsafe_rows_boxplot(combined_df, output_dir=output_dir),
        plot_total_unsafe_excess_histogram(combined_df, output_dir=output_dir),
        plot_max_utilization_ratio_histogram(combined_df, output_dir=output_dir),
    ]

    return outputs


def main() -> None:
    outputs = generate_all_stochastic_plots()

    print("\nGenerated stochastic validation figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()