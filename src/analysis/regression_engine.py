from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["synthetic", "literature_calibrated"]

TARGET_METRICS = [
    "total_unsafe_excess",
    "total_blocked_arrivals",
    "max_utilization_ratio",
    "num_unsafe_rows",
]

PREFERRED_POLICY_BASELINE_ORDER = [
    "optimized_network",
    "robust_optimized_network",
    "regime_robust_optimized_network",
    "myopic_milp",
    "no_transfer",
    "local_only",
    "no_control",
]

PREFERRED_SCENARIO_BASELINE_ORDER = [
    "baseline",
    "h3_icu_capacity_reduced",
    "transfer_disabled",
    "network_stress",
    "regional_crisis",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_replications(mode: str) -> pd.DataFrame:
    return safe_read_csv(RESULTS_DIR / f"scenario_suite_replications_{mode}.csv")


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def two_sided_pvalue(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))


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


def choose_baseline(values: list[str], preferred_order: list[str]) -> str:
    for item in preferred_order:
        if item in values:
            return item
    return sorted(values)[0]


def ordered_present(values: pd.Series, preferred_order: list[str]) -> list[str]:
    present = sorted(set(values.dropna().astype(str)))
    ordered = [v for v in preferred_order if v in present]
    extras = [v for v in present if v not in ordered]
    return ordered + extras


def build_design_matrix(
    df: pd.DataFrame,
    target_metric: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str], str, str]:
    work = df.copy()

    required = ["policy_name", "scenario", target_metric]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns for regression: {missing}")

    work = work.dropna(subset=["policy_name", "scenario", target_metric]).copy()
    work["policy_name"] = work["policy_name"].astype(str).str.strip()
    work["scenario"] = work["scenario"].astype(str).str.strip()
    work[target_metric] = pd.to_numeric(work[target_metric], errors="coerce")
    work = work.dropna(subset=[target_metric]).copy()

    present_policies = ordered_present(work["policy_name"], PREFERRED_POLICY_BASELINE_ORDER)
    present_scenarios = ordered_present(work["scenario"], PREFERRED_SCENARIO_BASELINE_ORDER)

    if len(present_policies) < 2:
        raise ValueError(f"Need at least 2 policies for regression on {target_metric}. Found: {present_policies}")
    if len(present_scenarios) < 2:
        raise ValueError(f"Need at least 2 scenarios for regression on {target_metric}. Found: {present_scenarios}")

    baseline_policy = choose_baseline(present_policies, PREFERRED_POLICY_BASELINE_ORDER)
    baseline_scenario = choose_baseline(present_scenarios, PREFERRED_SCENARIO_BASELINE_ORDER)

    work["policy_name"] = pd.Categorical(work["policy_name"], categories=present_policies, ordered=True)
    work["scenario"] = pd.Categorical(work["scenario"], categories=present_scenarios, ordered=True)

    policy_dummies = pd.get_dummies(work["policy_name"], prefix="policy", dtype=float)
    scenario_dummies = pd.get_dummies(work["scenario"], prefix="scenario", dtype=float)

    policy_base_col = f"policy_{baseline_policy}"
    scenario_base_col = f"scenario_{baseline_scenario}"

    X = pd.concat([policy_dummies, scenario_dummies], axis=1)
    X = X.drop(columns=[c for c in [policy_base_col, scenario_base_col] if c in X.columns], errors="ignore")

    # remove zero-variance columns if any slipped through
    nonconstant_cols = [c for c in X.columns if float(X[c].var()) > 1e-12]
    X = X[nonconstant_cols].copy()

    X.insert(0, "intercept", 1.0)
    X = X.astype(float)

    y = work[target_metric].to_numpy(dtype=float)
    feature_names = X.columns.tolist()

    return X, y, feature_names, baseline_policy, baseline_scenario


def fit_ols(X: np.ndarray, y: np.ndarray) -> dict:
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ X.T @ y

    yhat = X @ beta
    resid = y - yhat

    n = X.shape[0]
    k = X.shape[1]

    rss = float(resid.T @ resid)
    tss = float(((y - y.mean()) ** 2).sum())

    sigma2 = rss / max(n - k, 1)
    vcov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    z_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 1e-12)
    pvals = np.array([two_sided_pvalue(float(z)) if not np.isnan(z) else np.nan for z in z_stats])

    r2 = 1.0 - rss / tss if tss > 1e-12 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1) if pd.notna(r2) else np.nan

    return {
        "beta": beta,
        "se": se,
        "z": z_stats,
        "p": pvals,
        "rss": rss,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
    }


def build_coefficient_table(
    mode: str,
    metric: str,
    feature_names: list[str],
    fit: dict,
    baseline_policy: str,
    baseline_scenario: str,
) -> pd.DataFrame:
    rows = []
    for name, beta, se, z, p in zip(feature_names, fit["beta"], fit["se"], fit["z"], fit["p"]):
        rows.append(
            {
                "mode": mode,
                "metric": metric,
                "term": name,
                "coefficient": float(beta),
                "std_error": float(se),
                "z_stat": float(z) if not np.isnan(z) else np.nan,
                "p_value": float(p) if not np.isnan(p) else np.nan,
                "significance": significance_label(float(p)) if not np.isnan(p) else "NA",
                "baseline_policy": baseline_policy,
                "baseline_scenario": baseline_scenario,
            }
        )
    return pd.DataFrame(rows)


def build_fit_summary(
    mode: str,
    metric: str,
    fit: dict,
    baseline_policy: str,
    baseline_scenario: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mode": mode,
                "metric": metric,
                "n_obs": fit["n"],
                "n_params": fit["k"],
                "rss": fit["rss"],
                "r_squared": fit["r2"],
                "adj_r_squared": fit["adj_r2"],
                "baseline_policy": baseline_policy,
                "baseline_scenario": baseline_scenario,
            }
        ]
    )


def build_policy_effect_summary(coef_df: pd.DataFrame) -> pd.DataFrame:
    if coef_df.empty:
        return pd.DataFrame()

    policy_rows = coef_df[coef_df["term"].str.startswith("policy_")].copy()
    if policy_rows.empty:
        return pd.DataFrame()

    policy_rows["policy_name"] = policy_rows["term"].str.replace("policy_", "", regex=False)

    return policy_rows[
        [
            "mode",
            "metric",
            "policy_name",
            "coefficient",
            "std_error",
            "p_value",
            "significance",
            "baseline_policy",
            "baseline_scenario",
        ]
    ].sort_values(["mode", "metric", "coefficient"]).reset_index(drop=True)


def build_scenario_effect_summary(coef_df: pd.DataFrame) -> pd.DataFrame:
    if coef_df.empty:
        return pd.DataFrame()

    scenario_rows = coef_df[coef_df["term"].str.startswith("scenario_")].copy()
    if scenario_rows.empty:
        return pd.DataFrame()

    scenario_rows["scenario_name"] = scenario_rows["term"].str.replace("scenario_", "", regex=False)

    return scenario_rows[
        [
            "mode",
            "metric",
            "scenario_name",
            "coefficient",
            "std_error",
            "p_value",
            "significance",
            "baseline_policy",
            "baseline_scenario",
        ]
    ].sort_values(["mode", "metric", "coefficient"]).reset_index(drop=True)


def build_headline_table(coef_df: pd.DataFrame) -> pd.DataFrame:
    if coef_df.empty:
        return pd.DataFrame()

    wanted_terms = [
        "policy_robust_optimized_network",
        "policy_regime_robust_optimized_network",
    ]

    work = coef_df[
        (coef_df["metric"].isin(["total_unsafe_excess", "total_blocked_arrivals"]))
        & (coef_df["term"].isin(wanted_terms))
    ].copy()

    return work.sort_values(["mode", "metric", "term"]).reset_index(drop=True)


def run_one_mode(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_df = load_replications(mode)

    all_coefs = []
    all_fit = []

    for metric in TARGET_METRICS:
        X_df, y, feature_names, baseline_policy, baseline_scenario = build_design_matrix(rep_df, metric)
        X = X_df.to_numpy(dtype=float)

        fit = fit_ols(X, y)

        coef_df = build_coefficient_table(
            mode=mode,
            metric=metric,
            feature_names=feature_names,
            fit=fit,
            baseline_policy=baseline_policy,
            baseline_scenario=baseline_scenario,
        )
        fit_df = build_fit_summary(
            mode=mode,
            metric=metric,
            fit=fit,
            baseline_policy=baseline_policy,
            baseline_scenario=baseline_scenario,
        )

        all_coefs.append(coef_df)
        all_fit.append(fit_df)

    coef_out = pd.concat(all_coefs, ignore_index=True)
    fit_out = pd.concat(all_fit, ignore_index=True)

    coef_path = TABLES_DIR / f"final_regression_v2_coefficients_{mode}.csv"
    fit_path = TABLES_DIR / f"final_regression_v2_fit_summary_{mode}.csv"

    coef_out.to_csv(coef_path, index=False)
    fit_out.to_csv(fit_path, index=False)

    print(f"\n=== MODE: {mode} ===")
    print("\nRegression fit summary:")
    print(fit_out)

    print("\nSample coefficient rows:")
    print(coef_out.head(20))

    print("\nSaved:")
    print(coef_path)
    print(fit_path)

    return coef_out, fit_out


def main() -> None:
    all_coefs = []
    all_fit = []

    for mode in MODES:
        coef_df, fit_df = run_one_mode(mode)
        all_coefs.append(coef_df)
        all_fit.append(fit_df)

    coef_all = pd.concat(all_coefs, ignore_index=True)
    fit_all = pd.concat(all_fit, ignore_index=True)

    coef_all_path = TABLES_DIR / "final_regression_v2_coefficients_all_modes.csv"
    fit_all_path = TABLES_DIR / "final_regression_v2_fit_summary_all_modes.csv"
    coef_all.to_csv(coef_all_path, index=False)
    fit_all.to_csv(fit_all_path, index=False)

    policy_effects = build_policy_effect_summary(coef_all)
    scenario_effects = build_scenario_effect_summary(coef_all)
    headline = build_headline_table(coef_all)

    policy_path = TABLES_DIR / "final_regression_v2_policy_effects.csv"
    scenario_path = TABLES_DIR / "final_regression_v2_scenario_effects.csv"
    headline_path = TABLES_DIR / "final_regression_v2_headline.csv"

    policy_effects.to_csv(policy_path, index=False)
    scenario_effects.to_csv(scenario_path, index=False)
    headline.to_csv(headline_path, index=False)

    print("\n=== POLICY EFFECTS ===")
    print(policy_effects)

    print("\n=== SCENARIO EFFECTS ===")
    print(scenario_effects)

    print("\n=== HEADLINE TABLE ===")
    print(headline)

    print("\nSaved combined outputs:")
    print(coef_all_path)
    print(fit_all_path)
    print(policy_path)
    print(scenario_path)
    print(headline_path)


if __name__ == "__main__":
    main()