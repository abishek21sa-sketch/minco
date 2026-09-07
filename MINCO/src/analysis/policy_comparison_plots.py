from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_policy_summary(
    csv_path: str | Path = "results/policy_comparison_combined_summary.csv",
) -> pd.DataFrame:
    """
    Load the combined policy summary table.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(["scenario", "policy_name"]).reset_index(drop=True)
    return df


def _plot_grouped_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot grouped bars by scenario, with one bar per policy.
    """
    scenarios = list(df["scenario"].drop_duplicates())
    policies = list(df["policy_name"].drop_duplicates())

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = range(len(policies))
    width = 0.35

    for idx, scenario in enumerate(scenarios):
        sub = (
            df[df["scenario"] == scenario]
            .set_index("policy_name")
            .reindex(policies)
            .reset_index()
        )

        offsets = [i + (idx - (len(scenarios) - 1) / 2) * width for i in x]

        ax.bar(
            offsets,
            sub[value_col],
            width=width,
            label=scenario,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels(policies, rotation=20)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_policy_unsafe_excess(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot mean total unsafe excess by policy and scenario.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "policy_comparison_total_unsafe_excess.png"

    _plot_grouped_bar(
        df=df,
        value_col="total_unsafe_excess_mean",
        title="Mean Total Unsafe Excess by Policy",
        ylabel="Mean total unsafe excess",
        output_path=output_path,
    )
    return output_path


def plot_policy_max_utilization(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot mean max utilization ratio by policy and scenario.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "policy_comparison_max_utilization_ratio.png"

    _plot_grouped_bar(
        df=df,
        value_col="max_utilization_ratio_mean",
        title="Mean Max Utilization Ratio by Policy",
        ylabel="Mean max utilization ratio",
        output_path=output_path,
    )
    return output_path


def plot_policy_num_unsafe_rows(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot mean number of unsafe rows by policy and scenario.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "policy_comparison_num_unsafe_rows.png"

    _plot_grouped_bar(
        df=df,
        value_col="num_unsafe_rows_mean",
        title="Mean Unsafe Capacity Events by Policy",
        ylabel="Mean number of unsafe rows",
        output_path=output_path,
    )
    return output_path


def generate_all_policy_comparison_plots(
    csv_path: str | Path = "results/policy_comparison_combined_summary.csv",
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """
    Generate all standard policy-comparison plots.
    """
    df = _load_policy_summary(csv_path)

    outputs = [
        plot_policy_unsafe_excess(df, output_dir=output_dir),
        plot_policy_max_utilization(df, output_dir=output_dir),
        plot_policy_num_unsafe_rows(df, output_dir=output_dir),
    ]
    return outputs


def main() -> None:
    outputs = generate_all_policy_comparison_plots()

    print("\nGenerated policy comparison figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()