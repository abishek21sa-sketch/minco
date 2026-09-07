from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["synthetic", "literature_calibrated"]


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
    n_boot: int = 2000,
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
        n_boot=2000,
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
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "cohens_d": d,
    }


def build_statistical_tests(
    df: pd.DataFrame,
    mode: str,
    reference_policy: str = "optimized_network",
    comparators: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    if comparators is None:
        comparators = ["no_control", "local_only", "myopic_milp", "no_transfer"]

    if metrics is None:
        metrics = [
            "total_unsafe_excess",
            "max_utilization_ratio",
            "total_blocked_arrivals",
            "num_unsafe_rows",
        ]

    scenarios = sorted(df["scenario"].dropna().unique().tolist())

    rows = []
    counter = 0
    for scenario in scenarios:
        for metric in metrics:
            for comp in comparators:
                counter += 1
                rows.append(
                    compare_two_policies(
                        df=df,
                        mode=mode,
                        scenario=scenario,
                        metric=metric,
                        policy_a=reference_policy,
                        policy_b=comp,
                        bootstrap_seed=1000 + counter,
                    )
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["scenario", "metric", "approx_pvalue"]).reset_index(drop=True)


def summarize_significance_calls(test_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame()

    out = test_df.copy()

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

    out["significance"] = out["approx_pvalue"].apply(significance_label)
    out["effect_size_label"] = out["cohens_d"].apply(effect_label)

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
    cols = [c for c in cols if c in out.columns]
    return out[cols].copy()


def build_executive_tests_table(test_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame()

    working = summarize_significance_calls(test_df)

    # keep the most paper-relevant metrics
    keep_metrics = [
        "total_unsafe_excess",
        "max_utilization_ratio",
        "total_blocked_arrivals",
        "num_unsafe_rows",
    ]
    working = working[working["metric"].isin(keep_metrics)].copy()

    return working.sort_values(
        ["mode", "scenario", "metric", "approx_pvalue"]
    ).reset_index(drop=True)


def main() -> None:
    all_tests = []
    all_exec = []

    for mode in MODES:
        rep_df = load_replications(mode)

        tests = build_statistical_tests(
            df=rep_df,
            mode=mode,
            reference_policy="optimized_network",
            comparators=["no_control", "local_only", "myopic_milp", "no_transfer"],
            metrics=[
                "total_unsafe_excess",
                "max_utilization_ratio",
                "total_blocked_arrivals",
                "num_unsafe_rows",
            ],
        )

        exec_table = build_executive_tests_table(tests)

        tests_path = TABLES_DIR / f"statistical_tests_{mode}.csv"
        exec_path = TABLES_DIR / f"statistical_tests_executive_{mode}.csv"

        tests.to_csv(tests_path, index=False)
        exec_table.to_csv(exec_path, index=False)

        all_tests.append(tests)
        all_exec.append(exec_table)

        print(f"\n=== MODE: {mode} ===")
        print("\nExecutive statistical summary:")
        print(exec_table)

        print("\nSaved:")
        print(tests_path)
        print(exec_path)

    if all_tests:
        combined_tests = pd.concat(all_tests, ignore_index=True)
        combined_tests.to_csv(TABLES_DIR / "statistical_tests_all_modes.csv", index=False)

    if all_exec:
        combined_exec = pd.concat(all_exec, ignore_index=True)
        combined_exec.to_csv(TABLES_DIR / "statistical_tests_executive_all_modes.csv", index=False)

    print("\nSaved combined outputs:")
    print(TABLES_DIR / "statistical_tests_all_modes.csv")
    print(TABLES_DIR / "statistical_tests_executive_all_modes.csv")


if __name__ == "__main__":
    main()