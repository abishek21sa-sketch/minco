from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# FINAL STATISTICAL VALIDATION ENGINE
# For Regime-Robust Healthcare Policy Paper
# ==========================================================

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["synthetic", "literature_calibrated"]

PRIMARY_METRICS = [
    "total_unsafe_excess",
    "total_blocked_arrivals",
    "max_utilization_ratio",
    "num_unsafe_rows",
]

PRIMARY_POLICIES = [
    "optimized_network",
    "robust_optimized_network",
    "regime_robust_optimized_network",
]

BASELINE_COMPARATORS = [
    "no_control",
    "local_only",
    "myopic_milp",
    "no_transfer",
]


# ==========================================================
# Utilities
# ==========================================================

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def two_sided_pvalue(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_replications(mode: str) -> pd.DataFrame:
    return safe_read_csv(RESULTS_DIR / f"scenario_suite_replications_{mode}.csv")


# ==========================================================
# Effect Size
# ==========================================================

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled = ((len(x)-1)*vx + (len(y)-1)*vy) / (len(x)+len(y)-2)

    if pooled <= 1e-12:
        return 0.0

    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def effect_label(d: float) -> str:
    if pd.isna(d):
        return "NA"

    ad = abs(d)

    if ad >= 0.8:
        return "large"
    elif ad >= 0.5:
        return "medium"
    elif ad >= 0.2:
        return "small"
    else:
        return "negligible"


# ==========================================================
# Welch Approx Test
# ==========================================================

def welch_test(x: np.ndarray, y: np.ndarray):
    mx = np.mean(x)
    my = np.mean(y)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    se = np.sqrt(vx/len(x) + vy/len(y))

    if se <= 1e-12:
        return 0.0, 1.0

    z = (mx - my) / se
    p = two_sided_pvalue(z)

    return z, p


# ==========================================================
# Bootstrap CI
# ==========================================================

def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)

    vals = []

    for _ in range(n_boot):
        xb = rng.choice(x, len(x), replace=True)
        yb = rng.choice(y, len(y), replace=True)
        vals.append(np.mean(xb) - np.mean(yb))

    low = np.quantile(vals, alpha/2)
    high = np.quantile(vals, 1-alpha/2)

    return low, high


# ==========================================================
# Significance Labels
# ==========================================================

def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    else:
        return "ns"


# ==========================================================
# Compare Two Policies
# ==========================================================

def compare(
    df: pd.DataFrame,
    mode: str,
    scenario: str,
    metric: str,
    policy_a: str,
    policy_b: str,
):

    sdf = df[df["scenario"] == scenario]

    xa = (
        sdf[sdf["policy_name"] == policy_a][metric]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    xb = (
        sdf[sdf["policy_name"] == policy_b][metric]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    if len(xa) == 0 or len(xb) == 0:
        return None

    mean_a = float(np.mean(xa))
    mean_b = float(np.mean(xb))

    diff = mean_a - mean_b

    z, p = welch_test(xa, xb)
    ci_low, ci_high = bootstrap_ci(xa, xb)

    d = cohen_d(xa, xb)

    return {
        "mode": mode,
        "scenario": scenario,
        "metric": metric,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference_a_minus_b": diff,
        "z_stat": z,
        "p_value": p,
        "significance": sig_label(p),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohens_d": d,
        "effect_size": effect_label(d),
    }


# ==========================================================
# Full Engine
# ==========================================================

def run_mode(mode: str):

    df = load_replications(mode)

    scenarios = sorted(df["scenario"].unique())

    rows = []

    for scen in scenarios:
        for metric in PRIMARY_METRICS:

            # Main comparisons
            pairs = [
                ("regime_robust_optimized_network", "optimized_network"),
                ("regime_robust_optimized_network", "robust_optimized_network"),
                ("robust_optimized_network", "optimized_network"),
            ]

            for p in PRIMARY_POLICIES:
                for b in BASELINE_COMPARATORS:
                    pairs.append((p, b))

            for a, b in pairs:
                res = compare(df, mode, scen, metric, a, b)
                if res:
                    rows.append(res)

    out = pd.DataFrame(rows)

    out_path = TABLES_DIR / f"final_stat_validation_{mode}.csv"
    out.to_csv(out_path, index=False)

    print(f"\n=== MODE: {mode} ===")
    print(out.head(25))
    print(f"\nSaved: {out_path}")

    return out


# ==========================================================
# Executive Summary
# ==========================================================

def build_summary(df: pd.DataFrame):

    working = df.copy()

    working["policy_a"] = working["policy_a"].astype(str).str.strip()
    working["policy_b"] = working["policy_b"].astype(str).str.strip()
    working["metric"] = working["metric"].astype(str).str.strip()

    keep = working[
        (working["policy_a"] == "regime_robust_optimized_network")
        & (working["policy_b"] == "optimized_network")
        & (working["metric"] == "total_unsafe_excess")
    ].copy()

    if keep.empty:
        print("\nWARNING: No regime_robust_optimized_network vs optimized_network rows found.")
        print("Available policy_a values:", sorted(working["policy_a"].dropna().unique().tolist()))
        print("Available policy_b values:", sorted(working["policy_b"].dropna().unique().tolist()))
        print("Available metric values:", sorted(working["metric"].dropna().unique().tolist()))

        path = TABLES_DIR / "final_stat_summary.csv"
        keep.to_csv(path, index=False)
        print(f"\nSaved empty summary file for debugging: {path}")
        return

    keep["improved"] = keep["difference_a_minus_b"] < 0

    path = TABLES_DIR / "final_stat_summary.csv"
    keep.to_csv(path, index=False)

    print("\n=== FINAL HEADLINE SUMMARY ===")
    print(
        keep[
            [
                "mode",
                "scenario",
                "mean_a",
                "mean_b",
                "difference_a_minus_b",
                "p_value",
                "significance",
                "effect_size",
                "improved",
            ]
        ]
    )

    print(f"\nSaved: {path}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    all_frames = []

    for mode in MODES:
        out = run_mode(mode)
        all_frames.append(out)

    final_df = pd.concat(all_frames, ignore_index=True)

    all_path = TABLES_DIR / "final_stat_validation_all_modes.csv"
    final_df.to_csv(all_path, index=False)

    print(f"\nSaved combined file: {all_path}")

    build_summary(final_df)


if __name__ == "__main__":
    main()