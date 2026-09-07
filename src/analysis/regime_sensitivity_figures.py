from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = TABLES_DIR / "regime_sensitivity_summary.csv"
TORNADO_PATH = TABLES_DIR / "regime_sensitivity_tornado_input.csv"

POLICY_ORDER = [
    "regime_robust_optimized_network",
    "robust_optimized_network",
    "optimized_network",
]

DESIGN_ORDER = [
    "base",
    "icu_minus_20",
    "icu_plus_20",
    "transfer_minus_50",
    "transfer_plus_50",
    "demand_minus_20",
    "demand_plus_20",
    "normal_regime",
    "crisis_regime",
    "short_horizon",
    "long_horizon",
    "mild_uncertainty",
    "severe_uncertainty",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def ordered_designs(values: list[str]) -> list[str]:
    present = list(dict.fromkeys(values))
    ordered = [d for d in DESIGN_ORDER if d in present]
    extras = sorted([d for d in present if d not in ordered])
    return ordered + extras


def ordered_policies(values: list[str]) -> list[str]:
    present = list(dict.fromkeys(values))
    ordered = [p for p in POLICY_ORDER if p in present]
    extras = sorted([p for p in present if p not in ordered])
    return ordered + extras


# ============================================================
# Tornado charts
# ============================================================

def plot_tornado_chart(
    tornado_df: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str,
    filename: str,
) -> None:
    if tornado_df.empty or value_col not in tornado_df.columns:
        print(f"Skipping {filename}: missing tornado data.")
        return

    df = tornado_df.copy()
    df = df[df["design_name"] != "base"].copy()
    if df.empty:
        print(f"Skipping {filename}: no non-base rows.")
        return

    df = df.sort_values(value_col, ascending=True).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    y = range(len(df))
    x = df[value_col].astype(float).tolist()

    plt.barh(y, x)
    plt.yticks(y, df["design_name"].astype(str).tolist())
    plt.axvline(0.0, linewidth=1.2)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Base-case comparison
# ============================================================

def plot_base_case_policy_bars(
    summary_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    if summary_df.empty or metric_col not in summary_df.columns:
        print(f"Skipping {filename}: missing summary data.")
        return

    df = summary_df[summary_df["design_name"] == "base"].copy()
    if df.empty:
        print(f"Skipping {filename}: no base rows.")
        return

    df["policy_name"] = pd.Categorical(
        df["policy_name"],
        categories=ordered_policies(df["policy_name"].dropna().tolist()),
        ordered=True,
    )
    df = df.sort_values("policy_name")

    plt.figure(figsize=(8, 5))
    x = range(len(df))
    y = df[metric_col].astype(float).tolist()

    plt.bar(x, y)
    plt.xticks(list(x), df["policy_name"].astype(str).tolist(), rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Sensitivity scatter
# ============================================================

def plot_sensitivity_tradeoff_scatter(
    summary_df: pd.DataFrame,
    policy_name: str = "regime_robust_optimized_network",
    filename: str = "regime_sensitivity_tradeoff_scatter.png",
) -> None:
    if summary_df.empty:
        print(f"Skipping {filename}: empty summary.")
        return

    needed = {"design_name", "policy_name", "total_unsafe_excess_mean", "total_blocked_arrivals_mean"}
    if not needed.issubset(summary_df.columns):
        print(f"Skipping {filename}: missing required columns.")
        return

    df = summary_df[summary_df["policy_name"] == policy_name].copy()
    if df.empty:
        print(f"Skipping {filename}: no rows for policy {policy_name}.")
        return

    plt.figure(figsize=(10, 7))

    x = df["total_blocked_arrivals_mean"].astype(float).tolist()
    y = df["total_unsafe_excess_mean"].astype(float).tolist()

    plt.scatter(x, y, s=85)

    for _, row in df.iterrows():
        plt.annotate(
            str(row["design_name"]),
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
    plt.title("Sensitivity Tradeoff Frontier for Regime-Robust Policy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Regime comparison bars
# ============================================================

def plot_regime_comparison_bars(
    summary_df: pd.DataFrame,
    filename: str = "regime_sensitivity_regime_comparison.png",
) -> None:
    if summary_df.empty:
        print(f"Skipping {filename}: empty summary.")
        return

    needed = {"design_name", "policy_name", "total_unsafe_excess_mean"}
    if not needed.issubset(summary_df.columns):
        print(f"Skipping {filename}: missing required columns.")
        return

    regime_designs = ["normal_regime", "base", "crisis_regime"]
    df = summary_df[
        summary_df["design_name"].isin(regime_designs)
        & (summary_df["policy_name"] == "regime_robust_optimized_network")
    ].copy()

    if df.empty:
        print(f"Skipping {filename}: no regime rows.")
        return

    order = ["normal_regime", "base", "crisis_regime"]
    df["design_name"] = pd.Categorical(df["design_name"], categories=order, ordered=True)
    df = df.sort_values("design_name")

    plt.figure(figsize=(7, 5))
    x = range(len(df))
    y = df["total_unsafe_excess_mean"].astype(float).tolist()

    plt.bar(x, y)
    plt.xticks(list(x), df["design_name"].astype(str).tolist(), rotation=15, ha="right")
    plt.ylabel("Mean unsafe excess")
    plt.title("Regime Effect on Unsafe Excess")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Lever importance
# ============================================================

def classify_lever(design_name: str) -> str:
    if design_name.startswith("icu_"):
        return "ICU capacity"
    if design_name.startswith("transfer_"):
        return "Transfer capacity"
    if design_name.startswith("demand_"):
        return "Demand level"
    if "regime" in design_name:
        return "Regime state"
    if "horizon" in design_name:
        return "Lookahead horizon"
    if "uncertainty" in design_name:
        return "Uncertainty profile"
    return "Other"


def plot_lever_importance_bars(
    tornado_df: pd.DataFrame,
    filename: str = "regime_sensitivity_lever_importance.png",
) -> None:
    if tornado_df.empty:
        print(f"Skipping {filename}: empty tornado data.")
        return

    df = tornado_df.copy()
    df = df[df["design_name"] != "base"].copy()
    if df.empty:
        print(f"Skipping {filename}: no non-base rows.")
        return

    df["lever"] = df["design_name"].apply(classify_lever)
    df["abs_unsafe_change"] = df["unsafe_excess_change_vs_base"].abs()

    grouped = (
        df.groupby("lever", as_index=False)["abs_unsafe_change"]
        .mean()
        .sort_values("abs_unsafe_change", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(8, 5))
    x = range(len(grouped))
    y = grouped["abs_unsafe_change"].astype(float).tolist()

    plt.bar(x, y)
    plt.xticks(list(x), grouped["lever"].astype(str).tolist(), rotation=20, ha="right")
    plt.ylabel("Average absolute unsafe-excess shift")
    plt.title("Sensitivity Lever Importance")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Multi-policy design comparison
# ============================================================

def plot_designwise_policy_comparison(
    summary_df: pd.DataFrame,
    design_name: str,
    metric_col: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    if summary_df.empty or metric_col not in summary_df.columns:
        print(f"Skipping {filename}: missing summary data.")
        return

    df = summary_df[summary_df["design_name"] == design_name].copy()
    if df.empty:
        print(f"Skipping {filename}: no rows for design {design_name}.")
        return

    df["policy_name"] = pd.Categorical(
        df["policy_name"],
        categories=ordered_policies(df["policy_name"].dropna().tolist()),
        ordered=True,
    )
    df = df.sort_values("policy_name")

    plt.figure(figsize=(8, 5))
    x = range(len(df))
    y = df[metric_col].astype(float).tolist()

    plt.bar(x, y)
    plt.xticks(list(x), df["policy_name"].astype(str).tolist(), rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    summary_df = safe_read_csv(SUMMARY_PATH)
    tornado_df = safe_read_csv(TORNADO_PATH)

    plot_tornado_chart(
        tornado_df=tornado_df,
        value_col="unsafe_excess_change_vs_base",
        title="Tornado Chart: Unsafe Excess Sensitivity",
        xlabel="Change in unsafe excess vs base",
        filename="regime_sensitivity_tornado_unsafe.png",
    )

    plot_tornado_chart(
        tornado_df=tornado_df,
        value_col="blocked_arrivals_change_vs_base",
        title="Tornado Chart: Blocked Arrivals Sensitivity",
        xlabel="Change in blocked arrivals vs base",
        filename="regime_sensitivity_tornado_blocked.png",
    )

    plot_base_case_policy_bars(
        summary_df=summary_df,
        metric_col="total_unsafe_excess_mean",
        ylabel="Mean unsafe excess",
        title="Base Case Policy Comparison: Unsafe Excess",
        filename="regime_sensitivity_base_unsafe_bars.png",
    )

    plot_base_case_policy_bars(
        summary_df=summary_df,
        metric_col="total_blocked_arrivals_mean",
        ylabel="Mean blocked arrivals",
        title="Base Case Policy Comparison: Blocked Arrivals",
        filename="regime_sensitivity_base_blocked_bars.png",
    )

    plot_sensitivity_tradeoff_scatter(summary_df=summary_df)

    plot_regime_comparison_bars(summary_df=summary_df)

    plot_lever_importance_bars(tornado_df=tornado_df)

    plot_designwise_policy_comparison(
        summary_df=summary_df,
        design_name="demand_plus_20",
        metric_col="total_unsafe_excess_mean",
        ylabel="Mean unsafe excess",
        title="Demand +20%: Policy Comparison",
        filename="regime_sensitivity_demand_plus20_policy_bars.png",
    )

    plot_designwise_policy_comparison(
        summary_df=summary_df,
        design_name="icu_minus_20",
        metric_col="total_unsafe_excess_mean",
        ylabel="Mean unsafe excess",
        title="ICU Capacity -20%: Policy Comparison",
        filename="regime_sensitivity_icu_minus20_policy_bars.png",
    )

    print("\nSaved sensitivity figures to:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()