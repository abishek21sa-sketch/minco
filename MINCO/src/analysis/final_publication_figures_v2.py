from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


REGIME_SUMMARY_PATH = TABLES_DIR / "regime_suite_summary.csv"
REGIME_HEADLINE_PATH = TABLES_DIR / "regime_headline_summary.csv"
TRADEOFF_PATH = TABLES_DIR / "regime_tradeoff_frontier.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing: {path}", flush=True)
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"Loaded {path}: shape={df.shape}", flush=True)
    print(f"Columns: {df.columns.tolist()}", flush=True)
    return df


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}", flush=True)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_tradeoff_frontier(regime_summary: pd.DataFrame, tradeoff_df: pd.DataFrame) -> None:
    print("\nBuilding tradeoff frontier...", flush=True)

    df = tradeoff_df.copy() if not tradeoff_df.empty else regime_summary.copy()

    if df.empty:
        print("Skipping tradeoff frontier: no data.", flush=True)
        return

    x_col = find_col(
        df,
        [
            "total_blocked_arrivals_mean",
            "blocked_arrivals_mean",
            "blocked_regime",
            "blocked_arrivals",
            "mean_blocked_arrivals",
        ],
    )

    y_col = find_col(
        df,
        [
            "total_unsafe_excess_mean",
            "unsafe_excess_mean",
            "unsafe_regime",
            "unsafe_excess",
            "mean_unsafe_excess",
        ],
    )

    policy_col = find_col(df, ["policy_name", "policy", "controller", "method"])
    scenario_col = find_col(df, ["scenario", "design_name", "case"])

    if x_col is None or y_col is None:
        print("Skipping tradeoff frontier: missing x/y columns.", flush=True)
        print(f"Available columns: {df.columns.tolist()}", flush=True)
        return

    plot_df = df[[c for c in [scenario_col, policy_col, x_col, y_col] if c is not None]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        print("Skipping tradeoff frontier: x/y became empty after numeric conversion.", flush=True)
        return

    plt.figure(figsize=(11, 7))

    for _, row in plot_df.iterrows():
        x = row[x_col]
        y = row[y_col]

        if policy_col and scenario_col:
            label = f"{row[scenario_col]}\n{row[policy_col]}"
        elif policy_col:
            label = str(row[policy_col])
        elif scenario_col:
            label = str(row[scenario_col])
        else:
            label = ""

        plt.scatter(x, y, s=75)
        if label:
            plt.annotate(
                label,
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    plt.xlabel("Mean blocked arrivals")
    plt.ylabel("Mean unsafe excess")
    plt.title("Safety-Access Tradeoff Frontier")
    plt.grid(True, alpha=0.3)

    save_fig(FIGURES_DIR / "regime_publication_tradeoff_frontier.png")


def build_unsafe_heatmap(regime_summary: pd.DataFrame) -> None:
    print("\nBuilding unsafe heatmap...", flush=True)

    df = regime_summary.copy()

    if df.empty:
        print("Skipping unsafe heatmap: no data.", flush=True)
        return

    scenario_col = find_col(df, ["scenario", "design_name", "case"])
    policy_col = find_col(df, ["policy_name", "policy", "controller", "method"])
    unsafe_col = find_col(
        df,
        [
            "total_unsafe_excess_mean",
            "unsafe_excess_mean",
            "unsafe_regime",
            "unsafe_robust",
            "unsafe_nominal",
            "mean_unsafe_excess",
        ],
    )

    # Case 1: long format: scenario, policy_name, total_unsafe_excess_mean
    if scenario_col and policy_col and unsafe_col:
        heat = df.pivot_table(
            index=scenario_col,
            columns=policy_col,
            values=unsafe_col,
            aggfunc="mean",
        )

        preferred_cols = [
            "regime_robust_optimized_network",
            "robust_optimized_network",
            "optimized_network",
            "myopic_milp",
            "no_transfer",
            "local_only",
            "no_control",
        ]

        keep_cols = [c for c in preferred_cols if c in heat.columns]
        if keep_cols:
            heat = heat[keep_cols]

    # Case 2: wide format: scenario, unsafe_nominal, unsafe_robust, unsafe_regime
    elif scenario_col:
        wide_cols = [
            c for c in [
                "unsafe_nominal",
                "unsafe_robust",
                "unsafe_regime",
                "total_unsafe_excess_mean",
            ]
            if c in df.columns
        ]

        if not wide_cols:
            print("Skipping unsafe heatmap: no usable unsafe columns.", flush=True)
            print(f"Available columns: {df.columns.tolist()}", flush=True)
            return

        heat = df.set_index(scenario_col)[wide_cols].copy()

    else:
        print("Skipping unsafe heatmap: missing scenario column.", flush=True)
        print(f"Available columns: {df.columns.tolist()}", flush=True)
        return

    if heat.empty:
        print("Skipping unsafe heatmap: heatmap table empty.", flush=True)
        return

    heat = heat.apply(pd.to_numeric, errors="coerce")
    heat = heat.dropna(how="all")

    if heat.empty:
        print("Skipping unsafe heatmap: all heat values are NaN.", flush=True)
        return

    plt.figure(figsize=(11, 6))
    im = plt.imshow(heat.values, aspect="auto")

    plt.xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.values[i, j]
            label = "" if pd.isna(value) else f"{value:.2f}"
            plt.text(j, i, label, ha="center", va="center", fontsize=7)

    plt.colorbar(im, label="Mean unsafe excess")
    plt.title("Unsafe Excess Across Scenarios and Policies")

    save_fig(FIGURES_DIR / "regime_publication_unsafe_heatmap.png")


def build_regime_advantage_bars(regime_headline: pd.DataFrame) -> None:
    print("\nBuilding regime advantage bars...", flush=True)

    df = regime_headline.copy()

    if df.empty:
        print("Skipping advantage bars: no data.", flush=True)
        return

    scenario_col = find_col(df, ["scenario", "design_name", "case"])
    gain_col = find_col(
        df,
        [
            "regime_vs_nominal_gain",
            "unsafe_excess_gain",
            "unsafe_gain",
            "delta_regime_minus_nominal",
        ],
    )

    if scenario_col is None or gain_col is None:
        print("Skipping advantage bars: missing scenario/gain columns.", flush=True)
        print(f"Available columns: {df.columns.tolist()}", flush=True)
        return

    plot_df = df[[scenario_col, gain_col]].copy()
    plot_df[gain_col] = pd.to_numeric(plot_df[gain_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[gain_col])

    if plot_df.empty:
        print("Skipping advantage bars: no numeric gain values.", flush=True)
        return

    labels = plot_df[scenario_col].astype(str).tolist()
    vals = plot_df[gain_col].astype(float).tolist()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(vals)), vals)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("Unsafe excess reduction vs nominal")
    plt.title("Regime-Robust Advantage vs Nominal Optimization")

    for bar, value in zip(bars, vals):
        y = value + 0.03 if value >= 0 else value - 0.03
        va = "bottom" if value >= 0 else "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=8,
        )

    save_fig(FIGURES_DIR / "regime_publication_regime_advantage_bars.png")


def main() -> None:
    print("\nFINAL PUBLICATION FIGURES V2")
    print("=" * 70, flush=True)

    regime_summary = read_csv(REGIME_SUMMARY_PATH)
    regime_headline = read_csv(REGIME_HEADLINE_PATH)
    tradeoff = read_csv(TRADEOFF_PATH)

    build_tradeoff_frontier(regime_summary=regime_summary, tradeoff_df=tradeoff)
    build_unsafe_heatmap(regime_summary=regime_summary)
    build_regime_advantage_bars(regime_headline=regime_headline)

    print("\nPublication figure check:", flush=True)
    for filename in [
        "regime_publication_tradeoff_frontier.png",
        "regime_publication_unsafe_heatmap.png",
        "regime_publication_regime_advantage_bars.png",
    ]:
        path = FIGURES_DIR / filename
        print(f"{filename}: exists={path.exists()} | path={path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()