from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["synthetic", "literature_calibrated"]

DEFAULT_METRICS = [
    "total_unsafe_excess",
    "max_utilization_ratio",
    "num_unsafe_rows",
    "total_blocked_arrivals",
    "total_overflow_excess",
    "total_surge_gap",
]

PRIMARY_POLICIES = [
    "optimized_network",
    "robust_optimized_network",
]

COMPARATOR_POLICIES = [
    "no_control",
    "local_only",
    "myopic_milp",
    "no_transfer",
    "optimized_network",
    "robust_optimized_network",
]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def two_sided_pvalue_from_z(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_replications(mode: str) -> pd.DataFrame:
    return safe_read_csv(RESULTS_DIR / f"scenario_suite_replications_{mode}.csv")


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna().astype(float)
    y = y.dropna().astype(float)

    if len(x) < 2 or len(y) < 2:
        return 0.0

    mean_x = float(x.mean())
    mean_y = float(y.mean())
    var_x = float(x.var(ddof=1))
    var_y = float(y.var(ddof=1))

    pooled_var = ((len(x) - 1) * var_x + (len(y) - 1) * var_y) / (len(x) + len(y) - 2)
    if pooled_var <= 1e-12:
        return 0.0

    return (mean_x - mean_y) / sqrt(pooled_var)


def bootstrap_mean_difference_ci(
    x: pd.Series,
    y: pd.Series,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[float, float]:
    x = x.dropna().astype(float).to_numpy()
    y = y.dropna().astype(float).to_numpy()

    if len(x) == 0 or len(y) == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = float(np.mean(xb) - np.mean(yb))

    lower = float(np.quantile(diffs, alpha / 2.0))
    upper = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return lower, upper


def welch_style_test(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    x = x.dropna().astype(float)
    y = y.dropna().astype(float)

    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")

    mean_x = float(x.mean())
    mean_y = float(y.mean())
    var_x = float(x.var(ddof=1)) if len(x) > 1 else 0.0
    var_y = float(y.var(ddof=1)) if len(y) > 1 else 0.0

    se = sqrt(var_x / len(x) + var_y / len(y))
    if se <= 1e-12:
        return 0.0, 1.0

    z_like = (mean_x - mean_y) / se
    pval = two_sided_pvalue_from_z(z_like)
    return z_like, pval


def significance_label(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return "ns"


def effect_label(d: float) -> str:
    if pd.isna(d):
        return "NA"
    ad = abs(d)
    if ad >= 0.8:
        return "large"
    if ad >= 0.5:
        return "medium"
    if ad >= 0.2:
        return "small"
    return "negligible"


def compare_two_policies(
    df: pd.DataFrame,
    mode: str,
    scenario: str,
    metric: str,
    policy_a: str,
    policy_b: str,
    bootstrap_seed: int = 123,
) -> dict:
    sdf = df[df["scenario"] == scenario].copy()

    a = sdf[sdf["policy_name"] == policy_a][metric].dropna().reset_index(drop=True)
    b = sdf[sdf["policy_name"] == policy_b][metric].dropna().reset_index(drop=True)

    mean_a = float(a.mean()) if len(a) > 0 else float("nan")
    mean_b = float(b.mean()) if len(b) > 0 else float("nan")
    diff = mean_a - mean_b if pd.notna(mean_a) and pd.notna(mean_b) else float("nan")

    z_like, pval = welch_style_test(a, b)
    ci_low, ci_high = bootstrap_mean_difference_ci(
        a,
        b,
        n_boot=4000,
        alpha=0.05,
        seed=bootstrap_seed,
    )
    d = cohen_d(a, b)

    return {
        "mode": mode,
        "scenario": scenario,
        "metric": metric,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference_a_minus_b": diff,
        "z_like_stat": z_like,
        "approx_pvalue": pval,
        "significance": significance_label(pval),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "cohens_d": d,
        "effect_size_label": effect_label(d),
    }


def build_pairwise_tests(
    df: pd.DataFrame,
    mode: str,
    reference_policies: list[str] | None = None,
    comparator_policies: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    if reference_policies is None:
        reference_policies = PRIMARY_POLICIES
    if comparator_policies is None:
        comparator_policies = COMPARATOR_POLICIES
    if metrics is None:
        metrics = DEFAULT_METRICS

    scenarios = sorted(df["scenario"].dropna().unique().tolist())

    rows = []
    counter = 0

    for scenario in scenarios:
        for metric in metrics:
            for ref in reference_policies:
                for comp in comparator_policies:
                    if ref == comp:
                        continue
                    counter += 1
                    rows.append(
                        compare_two_policies(
                            df=df,
                            mode=mode,
                            scenario=scenario,
                            metric=metric,
                            policy_a=ref,
                            policy_b=comp,
                            bootstrap_seed=1000 + counter,
                        )
                    )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["mode", "scenario", "metric", "policy_a", "approx_pvalue"]
    ).reset_index(drop=True)


def build_policy_win_table(
    df: pd.DataFrame,
    mode: str,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    if metrics is None:
        metrics = DEFAULT_METRICS

    rows = []
    scenarios = sorted(df["scenario"].dropna().unique().tolist())

    for scenario in scenarios:
        sdf = df[df["scenario"] == scenario].copy()

        for metric in metrics:
            if metric not in sdf.columns:
                continue

            grouped = (
                sdf.groupby("policy_name", as_index=False)
                .agg(
                    mean_value=(metric, "mean"),
                    std_value=(metric, "std"),
                    n=("replication", "nunique"),
                )
                .sort_values("mean_value")
                .reset_index(drop=True)
            )

            grouped["rank"] = range(1, len(grouped) + 1)
            grouped["mode"] = mode
            grouped["scenario"] = scenario
            grouped["metric"] = metric
            rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out[
        ["mode", "scenario", "metric", "policy_name", "rank", "mean_value", "std_value", "n"]
    ].copy()


def build_executive_table(
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame()

    keep_metrics = [
        "total_unsafe_excess",
        "max_utilization_ratio",
        "total_blocked_arrivals",
        "num_unsafe_rows",
    ]

    working = test_df[test_df["metric"].isin(keep_metrics)].copy()

    preferred_rows = []

    for _, row in working.iterrows():
        a = row["policy_a"]
        b = row["policy_b"]

        if a == "robust_optimized_network" and b == "optimized_network":
            preferred_rows.append(True)
        elif a == "robust_optimized_network" and b in ["no_control", "local_only", "myopic_milp", "no_transfer"]:
            preferred_rows.append(True)
        elif a == "optimized_network" and b in ["no_control", "local_only", "myopic_milp", "no_transfer"]:
            preferred_rows.append(True)
        else:
            preferred_rows.append(False)

    working = working[pd.Series(preferred_rows, index=working.index)].copy()

    cols = [
        "mode",
        "scenario",
        "metric",
        "policy_a",
        "policy_b",
        "mean_a",
        "mean_b",
        "difference_a_minus_b",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "approx_pvalue",
        "significance",
        "cohens_d",
        "effect_size_label",
    ]
    cols = [c for c in cols if c in working.columns]
    return working[cols].sort_values(
        ["mode", "scenario", "metric", "policy_a", "approx_pvalue"]
    ).reset_index(drop=True)


def build_robust_nominal_focus_table(
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame()

    focus = test_df[
        (test_df["policy_a"] == "robust_optimized_network")
        & (test_df["policy_b"] == "optimized_network")
    ].copy()

    focus = focus.sort_values(["mode", "scenario", "metric"]).reset_index(drop=True)
    return focus


def build_policy_leaderboard_table(
    win_df: pd.DataFrame,
) -> pd.DataFrame:
    if win_df.empty:
        return pd.DataFrame()

    primary_metrics = [
        "total_unsafe_excess",
        "max_utilization_ratio",
        "total_blocked_arrivals",
        "num_unsafe_rows",
    ]
    working = win_df[win_df["metric"].isin(primary_metrics)].copy()

    leaders = working[working["rank"] == 1].copy()
    leaders = leaders.sort_values(["mode", "scenario", "metric"]).reset_index(drop=True)
    return leaders


def main() -> None:
    all_tests = []
    all_wins = []
    all_exec = []
    all_focus = []
    all_leaders = []

    for mode in MODES:
        rep_df = load_replications(mode)

        tests = build_pairwise_tests(
            df=rep_df,
            mode=mode,
            reference_policies=["optimized_network", "robust_optimized_network"],
            comparator_policies=[
                "no_control",
                "local_only",
                "myopic_milp",
                "no_transfer",
                "optimized_network",
                "robust_optimized_network",
            ],
            metrics=DEFAULT_METRICS,
        )

        wins = build_policy_win_table(rep_df, mode=mode, metrics=DEFAULT_METRICS)
        executive = build_executive_table(tests)
        robust_focus = build_robust_nominal_focus_table(tests)
        leaders = build_policy_leaderboard_table(wins)

        tests_path = TABLES_DIR / f"final_policy_significance_tests_{mode}.csv"
        exec_path = TABLES_DIR / f"final_policy_significance_executive_{mode}.csv"
        wins_path = TABLES_DIR / f"final_policy_win_table_{mode}.csv"
        focus_path = TABLES_DIR / f"final_policy_robust_vs_nominal_{mode}.csv"
        leaders_path = TABLES_DIR / f"final_policy_leaderboard_{mode}.csv"

        tests.to_csv(tests_path, index=False)
        executive.to_csv(exec_path, index=False)
        wins.to_csv(wins_path, index=False)
        robust_focus.to_csv(focus_path, index=False)
        leaders.to_csv(leaders_path, index=False)

        all_tests.append(tests)
        all_wins.append(wins)
        all_exec.append(executive)
        all_focus.append(robust_focus)
        all_leaders.append(leaders)

        print(f"\n=== MODE: {mode} ===")
        print("\nRobust vs nominal focus:")
        print(robust_focus)

        print("\nPolicy leaders:")
        print(leaders)

        print("\nSaved:")
        print(tests_path)
        print(exec_path)
        print(wins_path)
        print(focus_path)
        print(leaders_path)

    if all_tests:
        pd.concat(all_tests, ignore_index=True).to_csv(
            TABLES_DIR / "final_policy_significance_tests_all_modes.csv",
            index=False,
        )

    if all_wins:
        pd.concat(all_wins, ignore_index=True).to_csv(
            TABLES_DIR / "final_policy_win_table_all_modes.csv",
            index=False,
        )

    if all_exec:
        pd.concat(all_exec, ignore_index=True).to_csv(
            TABLES_DIR / "final_policy_significance_executive_all_modes.csv",
            index=False,
        )

    if all_focus:
        pd.concat(all_focus, ignore_index=True).to_csv(
            TABLES_DIR / "final_policy_robust_vs_nominal_all_modes.csv",
            index=False,
        )

    if all_leaders:
        pd.concat(all_leaders, ignore_index=True).to_csv(
            TABLES_DIR / "final_policy_leaderboard_all_modes.csv",
            index=False,
        )

    print("\nSaved combined outputs:")
    print(TABLES_DIR / "final_policy_significance_tests_all_modes.csv")
    print(TABLES_DIR / "final_policy_win_table_all_modes.csv")
    print(TABLES_DIR / "final_policy_significance_executive_all_modes.csv")
    print(TABLES_DIR / "final_policy_robust_vs_nominal_all_modes.csv")
    print(TABLES_DIR / "final_policy_leaderboard_all_modes.csv")


if __name__ == "__main__":
    main()