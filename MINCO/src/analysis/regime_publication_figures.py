from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

POLICY_ORDER = [
    "regime_robust_optimized_network",
    "robust_optimized_network",
    "optimized_network",
    "myopic_milp",
    "no_transfer",
    "local_only",
    "no_control",
]

SCENARIO_ORDER = [
    "baseline",
    "h3_icu_capacity_reduced",
    "transfer_disabled",
    "network_stress",
    "regional_crisis",
]


# ============================================================
# IO
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def ordered_policies(values: list[str]) -> list[str]:
    present = list(dict.fromkeys(values))
    ordered = [p for p in POLICY_ORDER if p in present]
    extras = sorted([p for p in present if p not in ordered])
    return ordered + extras


def ordered_scenarios(values: list[str]) -> list[str]:
    present = list(dict.fromkeys(values))
    ordered = [s for s in SCENARIO_ORDER if s in present]
    extras = sorted([s for s in present if s not in ordered])
    return ordered + extras


# ============================================================
# Figure 1: Tradeoff frontier
# ============================================================

def plot_regime_tradeoff_frontier(
    frontier_df: pd.DataFrame,
    filename: str = "regime_tradeoff_frontier.png",
) -> None:
    if frontier_df.empty:
        print("Skipping tradeoff frontier: no data.")
        return

    plt.figure(figsize=(10, 7))

    scenarios = ordered_scenarios(frontier_df["scenario"].dropna().tolist())

    for scenario in scenarios:
        sdf = frontier_df[frontier_df["scenario"] == scenario].copy()
        if sdf.empty:
            continue

        x = sdf["total_blocked_arrivals_mean"].astype(float).tolist()
        y = sdf["total_unsafe_excess_mean"].astype(float).tolist()

        plt.scatter(x, y, s=90, label=scenario)

        for _, row in sdf.iterrows():
            plt.annotate(
                str(row["policy_name"]),
                (
                    float(row["total_blocked_arrivals_mean"]),
                    float(row["total_unsafe_excess_mean"]),
                ),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    plt.xlabel("Mean blocked arrivals")
    plt.ylabel("Mean unsafe excess")
    plt.title("Safety vs Access Frontier for Regime-Aware Policies")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 2: Unsafe excess heatmap
# ============================================================

def plot_regime_unsafe_heatmap(
    summary_df: pd.DataFrame,
    filename: str = "regime_unsafe_heatmap.png",
) -> None:
    if summary_df.empty:
        print("Skipping unsafe heatmap: no data.")
        return

    pivot = summary_df.pivot_table(
        index="scenario",
        columns="policy_name",
        values="total_unsafe_excess_mean",
        aggfunc="first",
    )

    row_order = ordered_scenarios(pivot.index.tolist())
    col_order = ordered_policies(pivot.columns.tolist())

    pivot = pivot.reindex(index=row_order, columns=col_order)

    plt.figure(figsize=(10, 5.5))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Mean unsafe excess")
    plt.xticks(range(len(pivot.columns)), pivot.columns.tolist(), rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.tolist())
    plt.xlabel("Policy")
    plt.ylabel("Scenario")
    plt.title("Unsafe Excess Heatmap Across Scenarios and Policies")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 3: Regime advantage bars
# ============================================================

def plot_regime_advantage_bars(
    headline_df: pd.DataFrame,
    filename: str = "regime_advantage_bars.png",
) -> None:
    if headline_df.empty:
        print("Skipping regime advantage bars: no data.")
        return

    df = headline_df.copy()
    df["scenario"] = pd.Categorical(
        df["scenario"],
        categories=ordered_scenarios(df["scenario"].dropna().tolist()),
        ordered=True,
    )
    df = df.sort_values("scenario")

    plt.figure(figsize=(9, 6))
    x = range(len(df))
    y = df["regime_vs_nominal_gain"].astype(float).tolist()

    plt.bar(x, y)
    plt.xticks(list(x), df["scenario"].astype(str).tolist(), rotation=20, ha="right")
    plt.ylabel("Unsafe excess reduction vs nominal")
    plt.title("Regime-Robust Policy Advantage Over Nominal Optimized Policy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 4: Average rank bars
# ============================================================

def plot_average_rank_bars(
    avg_rank_df: pd.DataFrame,
    metric: str = "unsafe_excess",
    filename: str = "regime_average_rank_unsafe_excess.png",
) -> None:
    if avg_rank_df.empty:
        print("Skipping average rank bars: no data.")
        return

    sdf = avg_rank_df[avg_rank_df["metric"] == metric].copy()
    if sdf.empty:
        print(f"Skipping average rank bars: no rows for metric={metric}")
        return

    sdf["policy_name"] = pd.Categorical(
        sdf["policy_name"],
        categories=ordered_policies(sdf["policy_name"].dropna().tolist()),
        ordered=True,
    )
    sdf = sdf.sort_values("policy_name")

    plt.figure(figsize=(10, 6))
    x = range(len(sdf))
    y = sdf["avg_rank"].astype(float).tolist()

    plt.bar(x, y)
    plt.xticks(list(x), sdf["policy_name"].astype(str).tolist(), rotation=25, ha="right")
    plt.ylabel("Average rank")
    plt.title(f"Average Policy Rank for {metric.replace('_', ' ').title()}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 5: Policy win counts
# ============================================================

def plot_policy_win_counts(
    wins_df: pd.DataFrame,
    filename: str = "regime_policy_win_counts.png",
) -> None:
    if wins_df.empty:
        print("Skipping policy win counts: no data.")
        return

    wins_df = wins_df.copy()
    wins_df["policy_name"] = pd.Categorical(
        wins_df["policy_name"],
        categories=ordered_policies(wins_df["policy_name"].dropna().tolist()),
        ordered=True,
    )
    wins_df = wins_df.sort_values("policy_name")

    plt.figure(figsize=(10, 6))
    x = range(len(wins_df))
    y = wins_df["num_metric_wins"].astype(float).tolist()

    plt.bar(x, y)
    plt.xticks(list(x), wins_df["policy_name"].astype(str).tolist(), rotation=25, ha="right")
    plt.ylabel("Number of metric wins")
    plt.title("Policy Win Counts Across Regime-Aware Comparisons")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 6: Scenario panels
# ============================================================

def plot_regime_scenario_tradeoff_panels(
    frontier_df: pd.DataFrame,
    filename: str = "regime_scenario_tradeoff_panels.png",
) -> None:
    if frontier_df.empty:
        print("Skipping tradeoff panels: no data.")
        return

    scenarios = ordered_scenarios(frontier_df["scenario"].dropna().tolist())
    if not scenarios:
        print("Skipping tradeoff panels: no scenarios.")
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sdf = frontier_df[frontier_df["scenario"] == scenario].copy()

        if sdf.empty:
            ax.set_visible(False)
            continue

        ax.scatter(
            sdf["total_blocked_arrivals_mean"].astype(float),
            sdf["total_unsafe_excess_mean"].astype(float),
            s=75,
        )

        for _, row in sdf.iterrows():
            ax.annotate(
                str(row["policy_name"]),
                (
                    float(row["total_blocked_arrivals_mean"]),
                    float(row["total_unsafe_excess_mean"]),
                ),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_title(scenario)
        ax.set_xlabel("Blocked arrivals")
        ax.set_ylabel("Unsafe excess")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Scenario-Specific Safety vs Access Tradeoffs", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Figure 7: Relative improvement bars
# ============================================================

def plot_relative_improvement_bars(
    rel_imp_df: pd.DataFrame,
    filename: str = "regime_relative_improvement_bars.png",
) -> None:
    if rel_imp_df.empty:
        print("Skipping relative improvement bars: no data.")
        return

    sdf = rel_imp_df[
        (rel_imp_df["base_policy"] == "optimized_network")
        & (rel_imp_df["challenger_policy"] == "regime_robust_optimized_network")
    ].copy()

    if sdf.empty:
        print("Skipping relative improvement bars: no optimized vs regime-robust rows.")
        return

    sdf["scenario"] = pd.Categorical(
        sdf["scenario"],
        categories=ordered_scenarios(sdf["scenario"].dropna().tolist()),
        ordered=True,
    )
    sdf = sdf.sort_values("scenario")

    metrics = [
        "unsafe_excess_relative_improvement",
        "blocked_arrivals_relative_improvement",
        "max_utilization_relative_improvement",
        "overflow_relative_improvement",
    ]
    metrics = [m for m in metrics if m in sdf.columns]

    if not metrics:
        print("Skipping relative improvement bars: no usable metrics.")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        x = range(len(sdf))
        y = sdf[metric].astype(float).tolist()

        ax.bar(x, y)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sdf["scenario"].astype(str).tolist(), rotation=20, ha="right")
        ax.set_title(metric.replace("_", " ").replace("relative improvement", "improvement"))
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Relative Improvements of Regime-Robust Over Nominal Optimized Policy", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    summary_df = safe_read_csv(TABLES_DIR / "regime_suite_summary.csv")
    rankings_df = safe_read_csv(TABLES_DIR / "regime_policy_rankings.csv")
    wins_df = safe_read_csv(TABLES_DIR / "regime_policy_win_counts.csv")
    avg_rank_df = safe_read_csv(TABLES_DIR / "regime_policy_average_ranks.csv")
    frontier_df = safe_read_csv(TABLES_DIR / "regime_tradeoff_frontier.csv")
    rel_imp_df = safe_read_csv(TABLES_DIR / "regime_relative_improvements.csv")
    exec_df = safe_read_csv(TABLES_DIR / "regime_executive_summary.csv")
    headline_df = safe_read_csv(TABLES_DIR / "regime_headline_summary.csv")

    plot_regime_tradeoff_frontier(frontier_df)
    plot_regime_unsafe_heatmap(summary_df)
    plot_regime_advantage_bars(headline_df)
    plot_average_rank_bars(avg_rank_df, metric="unsafe_excess", filename="regime_average_rank_unsafe_excess.png")
    plot_average_rank_bars(avg_rank_df, metric="blocked_arrivals", filename="regime_average_rank_blocked_arrivals.png")
    plot_policy_win_counts(wins_df)
    plot_regime_scenario_tradeoff_panels(frontier_df)
    plot_relative_improvement_bars(rel_imp_df)

    print("\nSaved regime publication figures to:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()