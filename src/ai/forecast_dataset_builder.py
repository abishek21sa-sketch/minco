"""Legacy research dataset builder.

IMPORTANT: ``time_index`` in this module orders independent Monte Carlo
replications. It is not a calendar-time hospital operations axis and must not
be used as deployment-grade temporal forecasting evidence. The current
operational forecasting path is ``src.ai.demand_forecast_quantile``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
AI_DATA_DIR.mkdir(parents=True, exist_ok=True)

REGIME_REPLICATIONS_PATH = TABLES_DIR / "regime_suite_replications.csv"
SENSITIVITY_REPLICATIONS_PATH = TABLES_DIR / "regime_sensitivity_replications.csv"


# ============================================================
# IO
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# Regime labeling helpers
# ============================================================

def infer_regime_from_scenario(scenario: str) -> str:
    s = str(scenario).strip().lower()

    if s in {"baseline"}:
        return "normal"
    if s in {"h3_icu_capacity_reduced", "transfer_disabled", "network_stress"}:
        return "surge"
    if s in {"regional_crisis"}:
        return "crisis"

    return "normal"


def infer_regime_from_design(row: pd.Series) -> str:
    if "regime" in row.index and pd.notna(row["regime"]):
        return str(row["regime"]).strip().lower()

    if "design_name" in row.index:
        name = str(row["design_name"]).strip().lower()
        if "crisis" in name:
            return "crisis"
        if "surge" in name:
            return "surge"
        if "normal" in name:
            return "normal"

    return "normal"


def next_regime_label(current_regime: str) -> str:
    """
    Simple deterministic proxy target for v1 forecasting.
    You can later replace this with sampled regime transitions or Markov probabilities.
    """
    current_regime = str(current_regime).strip().lower()

    if current_regime == "normal":
        return "surge"
    if current_regime == "surge":
        return "crisis"
    return "crisis"


# ============================================================
# Core feature engineering
# ============================================================

def add_replication_time_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "replication" not in out.columns:
        out["replication"] = np.arange(len(out))

    sort_cols = []
    for c in ["scenario", "design_name", "policy_name", "replication"]:
        if c in out.columns:
            sort_cols.append(c)

    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    group_cols = [c for c in ["scenario", "design_name", "policy_name"] if c in out.columns]
    if not group_cols:
        group_cols = ["policy_name"] if "policy_name" in out.columns else ["replication"]

    out["time_index"] = out.groupby(group_cols).cumcount()

    return out


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "scenario" in out.columns:
        out["current_regime"] = out["scenario"].apply(infer_regime_from_scenario)
    elif "design_name" in out.columns or "regime" in out.columns:
        out["current_regime"] = out.apply(infer_regime_from_design, axis=1)
    else:
        out["current_regime"] = "normal"

    out["next_regime_label"] = out["current_regime"].apply(next_regime_label)
    return out


def add_risk_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    unsafe_col = first_existing(out, ["total_unsafe_excess", "unsafe_excess"])
    blocked_col = first_existing(out, ["total_blocked_arrivals", "blocked_arrivals"])
    util_col = first_existing(out, ["max_utilization_ratio", "max_utilization"])
    overflow_col = first_existing(out, ["total_overflow_excess", "overflow_excess"])
    unsafe_rows_col = first_existing(out, ["num_unsafe_rows", "unsafe_rows"])

    if unsafe_col is not None:
        out["unsafe_positive_flag"] = (pd.to_numeric(out[unsafe_col], errors="coerce").fillna(0.0) > 0).astype(int)
        out["unsafe_high_flag"] = (pd.to_numeric(out[unsafe_col], errors="coerce").fillna(0.0) > 5.0).astype(int)
    else:
        out["unsafe_positive_flag"] = 0
        out["unsafe_high_flag"] = 0

    if blocked_col is not None:
        out["blocked_positive_flag"] = (pd.to_numeric(out[blocked_col], errors="coerce").fillna(0.0) > 0).astype(int)
        out["blocked_high_flag"] = (pd.to_numeric(out[blocked_col], errors="coerce").fillna(0.0) > 5.0).astype(int)
    else:
        out["blocked_positive_flag"] = 0
        out["blocked_high_flag"] = 0

    if util_col is not None:
        util = pd.to_numeric(out[util_col], errors="coerce").fillna(0.0)
        out["utilization_high_flag"] = (util > 1.10).astype(int)
        out["utilization_critical_flag"] = (util > 1.30).astype(int)
    else:
        out["utilization_high_flag"] = 0
        out["utilization_critical_flag"] = 0

    if overflow_col is not None:
        out["overflow_positive_flag"] = (pd.to_numeric(out[overflow_col], errors="coerce").fillna(0.0) > 0).astype(int)
    else:
        out["overflow_positive_flag"] = 0

    if unsafe_rows_col is not None:
        out["unsafe_rows_positive_flag"] = (pd.to_numeric(out[unsafe_rows_col], errors="coerce").fillna(0.0) > 0).astype(int)
    else:
        out["unsafe_rows_positive_flag"] = 0

    return out


def add_lag_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()

    group_cols = [c for c in ["scenario", "design_name", "policy_name"] if c in out.columns]
    if not group_cols:
        group_cols = ["policy_name"] if "policy_name" in out.columns else ["replication"]

    for col in numeric_cols:
        if col not in out.columns:
            continue

        series = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[col] = series

        out[f"{col}_lag1"] = out.groupby(group_cols)[col].shift(1)
        out[f"{col}_lag2"] = out.groupby(group_cols)[col].shift(2)

        out[f"{col}_rollmean_3"] = (
            out.groupby(group_cols)[col]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
        out[f"{col}_rollmax_3"] = (
            out.groupby(group_cols)[col]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).max())
        )

    return out


def add_next_step_targets(
    df: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()

    group_cols = [c for c in ["scenario", "design_name", "policy_name"] if c in out.columns]
    if not group_cols:
        group_cols = ["policy_name"] if "policy_name" in out.columns else ["replication"]

    for col in target_cols:
        if col not in out.columns:
            continue
        out[f"target_next_{col}"] = out.groupby(group_cols)[col].shift(-1)

    return out


def encode_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "policy_name" in out.columns:
        out["policy_name"] = out["policy_name"].astype(str).str.strip()
    if "scenario" in out.columns:
        out["scenario"] = out["scenario"].astype(str).str.strip()
    if "design_name" in out.columns:
        out["design_name"] = out["design_name"].astype(str).str.strip()

    # simple numeric encodings for v1
    regime_map = {"normal": 0, "surge": 1, "crisis": 2}
    out["current_regime_code"] = out["current_regime"].map(regime_map).fillna(0).astype(int)

    if "lookahead_horizon" in out.columns:
        out["lookahead_horizon"] = pd.to_numeric(out["lookahead_horizon"], errors="coerce").fillna(0).astype(int)
    else:
        out["lookahead_horizon"] = 0

    for col in ["icu_capacity_scale", "transfer_capacity_scale", "demand_scale"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0)
        else:
            out[col] = 1.0

    return out


# ============================================================
# Dataset builders
# ============================================================

def build_forecast_dataset_from_replications(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = add_replication_time_index(out)
    out = add_regime_columns(out)
    out = add_risk_targets(out)

    numeric_core = [
        c for c in [
            "total_unsafe_excess",
            "total_blocked_arrivals",
            "max_utilization_ratio",
            "num_unsafe_rows",
            "total_overflow_excess",
            "total_surge_gap",
        ]
        if c in out.columns
    ]

    out = add_lag_features(out, numeric_cols=numeric_core)
    out = add_next_step_targets(out, target_cols=numeric_core)
    out = encode_context_features(out)

    # add next-step classification targets
    if "target_next_total_unsafe_excess" in out.columns:
        out["target_next_unsafe_positive_flag"] = (
            pd.to_numeric(out["target_next_total_unsafe_excess"], errors="coerce").fillna(0.0) > 0
        ).astype(int)

    if "target_next_total_blocked_arrivals" in out.columns:
        out["target_next_blocked_positive_flag"] = (
            pd.to_numeric(out["target_next_total_blocked_arrivals"], errors="coerce").fillna(0.0) > 0
        ).astype(int)

    if "target_next_max_utilization_ratio" in out.columns:
        out["target_next_utilization_critical_flag"] = (
            pd.to_numeric(out["target_next_max_utilization_ratio"], errors="coerce").fillna(0.0) > 1.2
        ).astype(int)

    # drop final rows per group where next-step targets do not exist
    target_cols = [c for c in out.columns if c.startswith("target_next_")]
    if target_cols:
        out = out.dropna(subset=target_cols, how="all").copy()

    # fill lag nulls after target dropping
    lag_cols = [c for c in out.columns if "_lag" in c or "_roll" in c]
    for c in lag_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out.reset_index(drop=True)


def build_regime_suite_forecast_dataset() -> pd.DataFrame:
    rep_df = safe_read_csv(REGIME_REPLICATIONS_PATH)
    if rep_df.empty:
        return pd.DataFrame()

    out = build_forecast_dataset_from_replications(rep_df)
    out["dataset_source"] = "regime_suite"
    return out


def build_sensitivity_forecast_dataset() -> pd.DataFrame:
    rep_df = safe_read_csv(SENSITIVITY_REPLICATIONS_PATH)
    if rep_df.empty:
        return pd.DataFrame()

    out = build_forecast_dataset_from_replications(rep_df)
    out["dataset_source"] = "regime_sensitivity"
    return out


def merge_forecast_datasets(
    regime_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = [df for df in [regime_df, sensitivity_df] if not df.empty]
    if not frames:
        return pd.DataFrame()

    # union columns safely
    all_cols = sorted(set().union(*[set(df.columns) for df in frames]))
    aligned = [df.reindex(columns=all_cols) for df in frames]

    out = pd.concat(aligned, ignore_index=True)
    return out.reset_index(drop=True)


# ============================================================
# Reports
# ============================================================

def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    row = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_sources": df["dataset_source"].nunique() if "dataset_source" in df.columns else 0,
        "n_policies": df["policy_name"].nunique() if "policy_name" in df.columns else 0,
        "n_scenarios": df["scenario"].nunique() if "scenario" in df.columns else 0,
        "n_designs": df["design_name"].nunique() if "design_name" in df.columns else 0,
    }

    for c in [
        "target_next_total_unsafe_excess",
        "target_next_total_blocked_arrivals",
        "target_next_max_utilization_ratio",
        "target_next_num_unsafe_rows",
        "target_next_unsafe_positive_flag",
        "target_next_blocked_positive_flag",
        "target_next_utilization_critical_flag",
    ]:
        if c in df.columns:
            row[f"non_null_{c}"] = int(df[c].notna().sum())

    return pd.DataFrame([row])


# ============================================================
# Main
# ============================================================

def main() -> None:
    regime_df = build_regime_suite_forecast_dataset()
    sensitivity_df = build_sensitivity_forecast_dataset()
    combined_df = merge_forecast_datasets(regime_df, sensitivity_df)
    summary_df = build_dataset_summary(combined_df)

    regime_path = AI_DATA_DIR / "forecast_dataset_regime_suite.csv"
    sensitivity_path = AI_DATA_DIR / "forecast_dataset_regime_sensitivity.csv"
    combined_path = AI_DATA_DIR / "forecast_dataset_combined.csv"
    summary_path = AI_DATA_DIR / "forecast_dataset_summary.csv"

    if not regime_df.empty:
        regime_df.to_csv(regime_path, index=False)
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(sensitivity_path, index=False)
    if not combined_df.empty:
        combined_df.to_csv(combined_path, index=False)
    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)

    print("\nFORECAST DATASET BUILDER")
    print("=" * 60)

    print("\nSaved:")
    if regime_df.empty:
        print(f"{regime_path} [not created: no source data]")
    else:
        print(regime_path)

    if sensitivity_df.empty:
        print(f"{sensitivity_path} [not created: no source data]")
    else:
        print(sensitivity_path)

    if combined_df.empty:
        print(f"{combined_path} [not created: no source data]")
    else:
        print(combined_path)

    if summary_df.empty:
        print(f"{summary_path} [not created: no source data]")
    else:
        print(summary_path)

    print("\nSummary:")
    print(summary_df if not summary_df.empty else "No data available.")


if __name__ == "__main__":
    main()