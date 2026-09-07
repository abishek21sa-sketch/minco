from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


def _build_pivot(
    df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """
    Build a pivot table for heatmap plotting.

    Rows:
        elective_rejection_cost
    Columns:
        generic_transfer_cost
    Values:
        selected metric
    """
    pivot = df.pivot(
        index="elective_rejection_cost",
        columns="generic_transfer_cost",
        values=value_col,
    )

    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def _plot_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    output_path: Path,
    value_fmt: str = ".2f",
) -> None:
    """
    Plot a simple annotated heatmap using matplotlib only.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    im = ax.imshow(pivot_df.values, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Generic transfer cost")
    ax.set_ylabel("Elective rejection cost")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([str(col) for col in pivot_df.columns])

    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([str(idx) for idx in pivot_df.index])

    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            ax.text(
                j,
                i,
                format(pivot_df.iloc[i, j], value_fmt),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_heatmap(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot heatmap of total ICU transfer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = _build_pivot(df, "total_icu_transfer")
    output_path = output_dir / "heatmap_total_icu_transfer.png"

    _plot_heatmap(
        pivot_df=pivot,
        title="Total ICU Transfer",
        output_path=output_path,
        value_fmt=".2f",
    )
    return output_path


def plot_rejection_heatmap(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot heatmap of total elective rejection.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = _build_pivot(df, "total_elective_rejected")
    output_path = output_dir / "heatmap_total_elective_rejected.png"

    _plot_heatmap(
        pivot_df=pivot,
        title="Total Elective Rejection",
        output_path=output_path,
        value_fmt=".2f",
    )
    return output_path


def plot_objective_heatmap(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot heatmap of objective value.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = _build_pivot(df, "objective_value")
    output_path = output_dir / "heatmap_objective_value.png"

    _plot_heatmap(
        pivot_df=pivot,
        title="Objective Value",
        output_path=output_path,
        value_fmt=".1f",
    )
    return output_path


def plot_accepted_electives_heatmap(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot heatmap of total elective accepted.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = _build_pivot(df, "total_elective_accepted")
    output_path = output_dir / "heatmap_total_elective_accepted.png"

    _plot_heatmap(
        pivot_df=pivot,
        title="Total Elective Accepted",
        output_path=output_path,
        value_fmt=".2f",
    )
    return output_path


def _build_policy_region_pivot(
    df: pd.DataFrame,
    transfer_tol: float = 1e-9,
    rejection_tol: float = 1e-9,
) -> pd.DataFrame:
    """
    Classify each sensitivity-grid cell into a policy region.

    Codes
    -----
    0 = Neutral
    1 = Transfer-dominant
    2 = Mixed
    3 = Rejection-dominant
    """
    classified = df.copy()

    def classify_row(row: pd.Series) -> int:
        transfer = float(row["total_icu_transfer"])
        rejection = float(row["total_elective_rejected"])

        has_transfer = transfer > transfer_tol
        has_rejection = rejection > rejection_tol

        if has_transfer and not has_rejection:
            return 1
        if has_transfer and has_rejection:
            return 2
        if (not has_transfer) and has_rejection:
            return 3
        return 0

    classified["policy_region_code"] = classified.apply(classify_row, axis=1)

    pivot = classified.pivot(
        index="elective_rejection_cost",
        columns="generic_transfer_cost",
        values="policy_region_code",
    )

    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def plot_policy_region_map(
    df: pd.DataFrame,
    output_dir: str | Path = "results/figures",
) -> Path:
    """
    Plot a categorical policy-region map.

    Region codes
    ------------
    0 = Neutral
    1 = Transfer-dominant
    2 = Mixed
    3 = Rejection-dominant
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = _build_policy_region_pivot(df)
    output_path = output_dir / "policy_region_map.png"

    fig, ax = plt.subplots(figsize=(8, 5.5))

    cmap = ListedColormap(["#dddddd", "#4daf4a", "#ffbf00", "#e41a1c"])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=3)

    ax.set_title("Policy Region Map")
    ax.set_xlabel("Generic transfer cost")
    ax.set_ylabel("Elective rejection cost")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])

    label_map = {
        0: "Neutral",
        1: "Transfer",
        2: "Mixed",
        3: "Reject",
    }

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            code = int(pivot.iloc[i, j])
            ax.text(
                j,
                i,
                label_map[code],
                ha="center",
                va="center",
            )

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Neutral", "Transfer", "Mixed", "Reject"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_all_sensitivity_plots(
    csv_path: str | Path = "results/sensitivity_results.csv",
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    """
    Generate all standard sensitivity-analysis plots from the results CSV.
    """
    df = pd.read_csv(csv_path)

    outputs = [
        plot_transfer_heatmap(df, output_dir=output_dir),
        plot_rejection_heatmap(df, output_dir=output_dir),
        plot_objective_heatmap(df, output_dir=output_dir),
        plot_accepted_electives_heatmap(df, output_dir=output_dir),
        plot_policy_region_map(df, output_dir=output_dir),
    ]
    return outputs


def main() -> None:
    outputs = generate_all_sensitivity_plots()

    print("\nGenerated figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()