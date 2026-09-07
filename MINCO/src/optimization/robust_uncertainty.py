from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class RobustArrivalConfig:
    default_rho: float = 0.10
    rho_by_hospital: dict[str, float] | None = None
    rho_by_hospital_cohort: dict[str, float] | None = None
    scenario_multiplier: float = 1.0
    min_arrival_floor: float = 0.0
    max_arrival_cap: float | None = None

    def __post_init__(self) -> None:
        if self.rho_by_hospital is None:
            self.rho_by_hospital = {}
        if self.rho_by_hospital_cohort is None:
            self.rho_by_hospital_cohort = {}


def _find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"None of the candidate columns {candidates} found in dataframe columns {list(df.columns)}"
    )


def _row_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def get_arrival_column(df: pd.DataFrame) -> str:
    return _find_first_existing(
        df,
        ["arrival_rate", "arrivals", "lambda", "arrival_volume", "mean_arrivals"],
    )


def get_hospital_column(df: pd.DataFrame) -> str:
    return _find_first_existing(df, ["hospital_id", "hospital"])


def get_cohort_column(df: pd.DataFrame) -> str:
    return _find_first_existing(df, ["cohort", "patient_cohort"])


def get_time_column(df: pd.DataFrame) -> str | None:
    for c in ["day", "time", "period", "t"]:
        if c in df.columns:
            return c
    return None


def resolve_rho_for_row(
    hospital: str,
    cohort: str,
    config: RobustArrivalConfig,
) -> float:
    hc_key = _row_key(hospital, cohort)

    if hc_key in config.rho_by_hospital_cohort:
        return float(config.rho_by_hospital_cohort[hc_key])

    if hospital in config.rho_by_hospital:
        return float(config.rho_by_hospital[hospital])

    return float(config.default_rho)


def build_robust_arrival_table(
    arrivals_df: pd.DataFrame,
    config: RobustArrivalConfig,
) -> pd.DataFrame:
    if arrivals_df.empty:
        return arrivals_df.copy()

    df = arrivals_df.copy()

    hospital_col = get_hospital_column(df)
    cohort_col = get_cohort_column(df)
    arrival_col = get_arrival_column(df)

    df[arrival_col] = pd.to_numeric(df[arrival_col], errors="coerce").fillna(0.0)

    rho_values = []
    robust_values = []

    for _, row in df.iterrows():
        hospital = str(row[hospital_col])
        cohort = str(row[cohort_col])

        rho = resolve_rho_for_row(hospital, cohort, config)
        base_arrival = float(row[arrival_col])

        robust_arrival = base_arrival * (1.0 + rho) * float(config.scenario_multiplier)
        robust_arrival = max(robust_arrival, float(config.min_arrival_floor))

        if config.max_arrival_cap is not None:
            robust_arrival = min(robust_arrival, float(config.max_arrival_cap))

        rho_values.append(rho)
        robust_values.append(robust_arrival)

    df["robust_rho"] = rho_values
    df["nominal_arrival"] = df[arrival_col].astype(float)
    df["robust_arrival"] = robust_values
    df["robust_delta"] = df["robust_arrival"] - df["nominal_arrival"]
    df["robust_multiplier"] = df["robust_arrival"] / df["nominal_arrival"].replace(0, pd.NA)

    return df


def apply_robust_arrivals_to_input(
    arrivals_df: pd.DataFrame,
    robust_arrival_table: pd.DataFrame,
    target_column_name: str | None = None,
) -> pd.DataFrame:
    if arrivals_df.empty:
        return arrivals_df.copy()

    out = arrivals_df.copy()

    arrival_col = get_arrival_column(out)
    if target_column_name is None:
        target_column_name = arrival_col

    merge_keys = []
    for c in ["hospital_id", "hospital", "cohort", "patient_cohort", "day", "time", "period", "t"]:
        if c in out.columns and c in robust_arrival_table.columns:
            merge_keys.append(c)

    if not merge_keys:
        raise ValueError("No common keys found to merge robust arrival table into arrivals input.")

    rhs = robust_arrival_table[merge_keys + ["robust_arrival"]].copy()

    out = out.merge(rhs, on=merge_keys, how="left", validate="one_to_one")

    if out["robust_arrival"].isnull().any():
        raise ValueError("Robust arrival table did not align correctly with arrivals input.")

    out[target_column_name] = out["robust_arrival"]
    out = out.drop(columns=["robust_arrival"])

    return out


def summarize_robust_arrivals(
    robust_arrival_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if robust_arrival_table.empty:
        return {
            "overall": pd.DataFrame(),
            "by_hospital": pd.DataFrame(),
            "by_hospital_cohort": pd.DataFrame(),
        }

    hospital_col = get_hospital_column(robust_arrival_table)
    cohort_col = get_cohort_column(robust_arrival_table)

    overall = pd.DataFrame(
        [
            {
                "nominal_total": float(robust_arrival_table["nominal_arrival"].sum()),
                "robust_total": float(robust_arrival_table["robust_arrival"].sum()),
                "delta_total": float(robust_arrival_table["robust_delta"].sum()),
                "mean_rho": float(robust_arrival_table["robust_rho"].mean()),
            }
        ]
    )

    by_hospital = (
        robust_arrival_table.groupby(hospital_col, as_index=False)
        .agg(
            nominal_total=("nominal_arrival", "sum"),
            robust_total=("robust_arrival", "sum"),
            delta_total=("robust_delta", "sum"),
            mean_rho=("robust_rho", "mean"),
        )
        .sort_values("delta_total", ascending=False)
        .reset_index(drop=True)
    )

    by_hospital_cohort = (
        robust_arrival_table.groupby([hospital_col, cohort_col], as_index=False)
        .agg(
            nominal_total=("nominal_arrival", "sum"),
            robust_total=("robust_arrival", "sum"),
            delta_total=("robust_delta", "sum"),
            mean_rho=("robust_rho", "mean"),
        )
        .sort_values("delta_total", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "overall": overall,
        "by_hospital": by_hospital,
        "by_hospital_cohort": by_hospital_cohort,
    }


def build_stress_test_configs() -> dict[str, RobustArrivalConfig]:
    return {
        "mild": RobustArrivalConfig(
            default_rho=0.05,
            scenario_multiplier=1.00,
        ),
        "moderate": RobustArrivalConfig(
            default_rho=0.10,
            scenario_multiplier=1.05,
        ),
        "severe": RobustArrivalConfig(
            default_rho=0.20,
            scenario_multiplier=1.10,
        ),
        "icu_heavy": RobustArrivalConfig(
            default_rho=0.08,
            rho_by_hospital_cohort={
                "H1|c3": 0.18,
                "H2|c3": 0.20,
                "H3|c3": 0.25,
            },
            scenario_multiplier=1.08,
        ),
        "h3_stress": RobustArrivalConfig(
            default_rho=0.08,
            rho_by_hospital={
                "H3": 0.25,
            },
            scenario_multiplier=1.05,
        ),
    }


def make_robustified_arrivals(
    arrivals_df: pd.DataFrame,
    config: RobustArrivalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    robust_table = build_robust_arrival_table(arrivals_df, config)
    robust_arrivals_df = apply_robust_arrivals_to_input(arrivals_df, robust_table)
    summary = summarize_robust_arrivals(robust_table)
    return robust_arrivals_df, robust_table, summary


def print_robust_summary(summary: dict[str, pd.DataFrame]) -> None:
    if not summary:
        print("No robust summary available.")
        return

    if "overall" in summary and not summary["overall"].empty:
        print("\nROBUST ARRIVAL SUMMARY: OVERALL")
        print(summary["overall"])

    if "by_hospital" in summary and not summary["by_hospital"].empty:
        print("\nROBUST ARRIVAL SUMMARY: BY HOSPITAL")
        print(summary["by_hospital"])

    if "by_hospital_cohort" in summary and not summary["by_hospital_cohort"].empty:
        print("\nROBUST ARRIVAL SUMMARY: BY HOSPITAL-COHORT")
        print(summary["by_hospital_cohort"])