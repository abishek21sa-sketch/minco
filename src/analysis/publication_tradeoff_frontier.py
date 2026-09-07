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
            "leaderboard": safe_read_csv(TABLES_DIR / f"final_policy_leaderboard_{mode}.csv"),
            "significance_exec": safe_read_csv(
                TABLES_DIR / f"final_policy_significance_executive_{mode}.csv"
            ),
            "robust_vs_nominal": safe_read_csv(
                TABLES_DIR / f"final_policy_robust_vs_nominal_{mode}.csv"
            ),
        }
    return data


def ordered_policies(df: pd.DataFrame, policy_col: str = "policy_name") -> list[str]:
    present = df[policy_col].dropna().unique().tolist()
    ordered = [p for p in POLICY_ORDER if p in present]
    extras = sorted([p for p in present if p not in ordered])
    return ordered + extras


def ordered_scenarios(df: pd.DataFrame, scenario_col: str = "scenario") -> list[str]:
    present = df[scenario_col].dropna().unique().tolist()
    ordered = [s for s in SCENARIO_ORDER if s in present]
    extras = sorted([s for s in present if s not in ordered])
    return ordered + extras


def plot_safety_access_frontier(
    summary_df: pd.DataFrame,
    mode: str,
    filename: str,
) -> None:
    if summary_df.empty:
        print(f"Skipping {filename}: empty summary")
        return

    plt.figure(figsize=(10, 7))

    for scenario in ordered_scenarios(summary_df):
        sdf = summary_df[summary_df["scenario"] == scenario].copy()
        if sdf.empty:
            continue

        x = sdf["total_blocked_arrivals_mean"].astype(float).tolist()
        y = sdf["total_unsafe_excess_mean"].astype(float).tolist()

        plt.scatter(x, y, s=80, label=scenario)

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
    plt.title(f"Safety vs Access Frontier | {mode}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_safety_access_frontier_single_scenario(
    summary_df: pd.DataFrame,
    mode: str,
    scenario: str,
    filename: str,
) -> None:
    sdf = summary_df[summary_df["scenario"] == scenario].copy()
    if sdf.empty:
        print(f"Skipping {filename}: no rows for scenario={scenario}")
        return

    order = ordered_policies(sdf)
    sdf["policy_name"] = pd.Categorical(sdf["policy_name"], categories=order, ordered=True)
    sdf = sdf.sort_values("policy_name")

    plt.figure(figsize=(9, 6))

    x = sdf["total_blocked_arrivals_mean"].astype(float).tolist()
    y = sdf["total_unsafe_excess_mean"].astype(float).tolist()

    plt.scatter(x, y, s=90)

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
    plt.title(f"Safety vs Access Frontier | {scenario} | {mode}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_robust_vs_nominal_scatter(
    focus_df: pd.DataFrame,
    mode: str,
    filename: str,
) -> None:
    if focus_df.empty:
        print(f"Skipping {filename}: empty robust-vs-nominal focus")
        return

    metrics = [
        "total_unsafe_excess",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_blocked_arrivals",
    ]

    plt.figure(figsize=(10, 7))

    xs = []
    ys = []
    labels = []

    for metric in metrics:
        mdf = focus_df[focus_df["metric"] == metric].copy()
        for _, row in mdf.iterrows():
            xs.append(float(row["mean_b"]))
            ys.append(float(row["mean_a"]))
            labels.append(f"{row['scenario']} | {metric}")

    if not xs:
        print(f"Skipping {filename}: no usable rows")
        return

    plt.scatter(xs, ys, s=85)

    lo = min(xs + ys)
    hi = max(xs + ys)
    plt.plot([lo, hi], [lo, hi], linewidth=1.2)

    for x, y, label in zip(xs, ys, labels):
        plt.annotate(
            label,
            (x, y),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel("Nominal optimized mean")
    plt.ylabel("Robust optimized mean")
    plt.title(f"Robust vs Nominal Policy Comparison | {mode}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_policy_win_heatmap(
    leaderboard_df: pd.DataFrame,
    mode: str,
    filename: str,
) -> None:
    if leaderboard_df.empty:
        print(f"Skipping {filename}: empty leaderboard")
        return

    working = leaderboard_df.copy()
    working["win_flag"] = 1

    pivot = (
        working.pivot_table(
            index="policy_name",
            columns="metric",
            values="win_flag",
            aggfunc="sum",
            fill_value=0,
        )
        .astype(float)
    )

    row_order = [p for p in POLICY_ORDER if p in pivot.index] + [p for p in pivot.index if p not in POLICY_ORDER]
    pivot = pivot.reindex(row_order)

    plt.figure(figsize=(9, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Number of scenario wins")
    plt.xticks(range(len(pivot.columns)), pivot.columns.tolist(), rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.tolist())
    plt.xlabel("Metric")
    plt.ylabel("Policy")
    plt.title(f"Policy Win Count Heatmap | {mode}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scenario_tradeoff_panels(
    summary_df: pd.DataFrame,
    mode: str,
    filename: str,
) -> None:
    scenarios = ordered_scenarios(summary_df)
    if not scenarios:
        print(f"Skipping {filename}: no scenarios")
        return

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sdf = summary_df[summary_df["scenario"] == scenario].copy()
        if sdf.empty:
            ax.set_visible(False)
            continue

        x = sdf["total_blocked_arrivals_mean"].astype(float).tolist()
        y = sdf["total_unsafe_excess_mean"].astype(float).tolist()

        ax.scatter(x, y, s=80)

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

    fig.suptitle(f"Scenario Tradeoff Panels | {mode}", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_frontier_gap_bars(
    summary_df: pd.DataFrame,
    mode: str,
    filename: str,
) -> None:
    if summary_df.empty:
        print(f"Skipping {filename}: empty summary")
        return

    rows = []

    for scenario in ordered_scenarios(summary_df):
        sdf = summary_df[summary_df["scenario"] == scenario].copy()
        if sdf.empty:
            continue

        for policy in ["optimized_network", "robust_optimized_network"]:
            ps = sdf[sdf["policy_name"] == policy]
            if ps.empty:
                continue
            row = ps.iloc[0]
            rows.append(
                {
                    "scenario": scenario,
                    "policy_name": policy,
                    "unsafe_excess": float(row["total_unsafe_excess_mean"]),
                    "blocked_arrivals": float(row["total_blocked_arrivals_mean"]),
                }
            )

    frontier_df = pd.DataFrame(rows)
    if frontier_df.empty:
        print(f"Skipping {filename}: no frontier comparison rows")
        return

    plt.figure(figsize=(10, 6))

    x = range(len(frontier_df))
    y = frontier_df["unsafe_excess"].tolist()

    plt.bar(x, y)
    plt.xticks(
        list(x),
        [
            f"{r['scenario']}\n{r['policy_name']}"
            for _, r in frontier_df.iterrows()
        ],
        rotation=20,
        ha="right",
    )
    plt.ylabel("Mean unsafe excess")
    plt.title(f"Unsafe Excess for Nominal vs Robust Policies | {mode}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    all_data = load_inputs()

    for mode, data in all_data.items():
        summary_df = data["scenario_summary"]
        leaderboard_df = data["leaderboard"]
        robust_focus_df = data["robust_vs_nominal"]

        plot_safety_access_frontier(
            summary_df=summary_df,
            mode=mode,
            filename=f"pub_tradeoff_frontier_all_{mode}.png",
        )

        for scenario in ordered_scenarios(summary_df):
            plot_safety_access_frontier_single_scenario(
                summary_df=summary_df,
                mode=mode,
                scenario=scenario,
                filename=f"pub_tradeoff_frontier_{scenario}_{mode}.png",
            )

        plot_robust_vs_nominal_scatter(
            focus_df=robust_focus_df,
            mode=mode,
            filename=f"pub_robust_vs_nominal_scatter_{mode}.png",
        )

        plot_policy_win_heatmap(
            leaderboard_df=leaderboard_df,
            mode=mode,
            filename=f"pub_policy_win_heatmap_{mode}.png",
        )

        plot_scenario_tradeoff_panels(
            summary_df=summary_df,
            mode=mode,
            filename=f"pub_scenario_tradeoff_panels_{mode}.png",
        )

        plot_frontier_gap_bars(
            summary_df=summary_df,
            mode=mode,
            filename=f"pub_nominal_vs_robust_unsafe_bars_{mode}.png",
        )

    print("\nSaved tradeoff frontier figures to:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()