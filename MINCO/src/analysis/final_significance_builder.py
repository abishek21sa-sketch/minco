from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
MANUSCRIPT_DIR = RESULTS_DIR / "manuscript_package_v2"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    TABLES_DIR / "regime_final_policy_significance_tests.csv",
    TABLES_DIR / "final_policy_significance_tests_all_modes.csv",
    TABLES_DIR / "final_stat_validation_all_modes.csv",
]

OUT_TABLES_PATH = TABLES_DIR / "regime_final_policy_scenario_headline.csv"
OUT_MANUSCRIPT_PATH = MANUSCRIPT_DIR / "T4_policy_significance_headline.csv"


PRIMARY_METRIC = "total_unsafe_excess"

PRIMARY_COMPARISONS = [
    ("regime_robust_optimized_network", "optimized_network"),
    ("regime_robust_optimized_network", "robust_optimized_network"),
    ("robust_optimized_network", "optimized_network"),
]

SCENARIO_ORDER = [
    "baseline",
    "h3_icu_capacity_reduced",
    "transfer_disabled",
    "network_stress",
    "regional_crisis",
]


def safe_read_first_existing(paths: list[Path]) -> tuple[pd.DataFrame, Path | None]:
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                return df, path
    return pd.DataFrame(), None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {
        "p_value": "approx_pvalue",
        "ci_low": "bootstrap_ci_low",
        "ci_high": "bootstrap_ci_high",
        "cohens_d": "cohens_d",
        "effect_size": "effect_size_label",
    }

    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    if "difference_a_minus_b" not in out.columns:
        if {"mean_a", "mean_b"}.issubset(out.columns):
            out["difference_a_minus_b"] = out["mean_a"] - out["mean_b"]

    return out


def scenario_sort_key(scenario: str) -> int:
    if scenario in SCENARIO_ORDER:
        return SCENARIO_ORDER.index(scenario)
    return len(SCENARIO_ORDER) + 1


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


def build_headline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = normalize_columns(df)

    required = ["scenario", "metric", "policy_a", "policy_b"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work["metric"] = work["metric"].astype(str)
    work["policy_a"] = work["policy_a"].astype(str)
    work["policy_b"] = work["policy_b"].astype(str)
    work["scenario"] = work["scenario"].astype(str)

    work = work[work["metric"] == PRIMARY_METRIC].copy()

    rows = []

    for policy_a, policy_b in PRIMARY_COMPARISONS:
        pair_df = work[
            (work["policy_a"] == policy_a)
            & (work["policy_b"] == policy_b)
        ].copy()

        for _, row in pair_df.iterrows():
            diff = float(row.get("difference_a_minus_b", np.nan))
            pval = float(row.get("approx_pvalue", np.nan))
            d = float(row.get("cohens_d", np.nan))

            rows.append(
                {
                    "scenario": row["scenario"],
                    "comparison": f"{policy_a} vs {policy_b}",
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "metric": PRIMARY_METRIC,
                    "mean_policy_a": row.get("mean_a", np.nan),
                    "mean_policy_b": row.get("mean_b", np.nan),
                    "difference_a_minus_b": diff,
                    "interpretation": (
                        "policy_a_better"
                        if diff < 0
                        else "policy_b_better"
                        if diff > 0
                        else "tie"
                    ),
                    "bootstrap_ci_low": row.get("bootstrap_ci_low", np.nan),
                    "bootstrap_ci_high": row.get("bootstrap_ci_high", np.nan),
                    "p_value": pval,
                    "significance": row.get("significance", significance_label(pval)),
                    "cohens_d": d,
                    "effect_size_label": row.get("effect_size_label", effect_label(d)),
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["scenario_sort"] = out["scenario"].apply(scenario_sort_key)
    out = out.sort_values(
        ["scenario_sort", "comparison"]
    ).drop(columns=["scenario_sort"]).reset_index(drop=True)

    return out


def build_compact_summary(headline_df: pd.DataFrame) -> pd.DataFrame:
    if headline_df.empty:
        return pd.DataFrame()

    rows = []

    for comparison in headline_df["comparison"].dropna().unique():
        sdf = headline_df[headline_df["comparison"] == comparison].copy()

        rows.append(
            {
                "comparison": comparison,
                "n_scenarios": len(sdf),
                "policy_a_better_count": int((sdf["interpretation"] == "policy_a_better").sum()),
                "policy_b_better_count": int((sdf["interpretation"] == "policy_b_better").sum()),
                "significant_at_0_05_count": int((pd.to_numeric(sdf["p_value"], errors="coerce") < 0.05).sum()),
                "mean_difference_a_minus_b": float(pd.to_numeric(sdf["difference_a_minus_b"], errors="coerce").mean()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df, source_path = safe_read_first_existing(INPUT_CANDIDATES)

    print("\nFINAL SIGNIFICANCE HEADLINE BUILDER")
    print("=" * 70)

    if df.empty:
        print("No usable significance input file found.")
        print("Checked:")
        for p in INPUT_CANDIDATES:
            print(p)
        return

    headline = build_headline(df)
    compact = build_compact_summary(headline)

    compact_path = TABLES_DIR / "regime_final_policy_significance_compact.csv"

    headline.to_csv(OUT_TABLES_PATH, index=False)
    headline.to_csv(OUT_MANUSCRIPT_PATH, index=False)
    compact.to_csv(compact_path, index=False)

    print(f"\nSource used: {source_path}")

    print("\n=== Headline significance table ===")
    print(headline.to_string(index=False))

    print("\n=== Compact summary ===")
    print(compact.to_string(index=False))

    print("\nSaved:")
    print(OUT_TABLES_PATH)
    print(OUT_MANUSCRIPT_PATH)
    print(compact_path)


if __name__ == "__main__":
    main()