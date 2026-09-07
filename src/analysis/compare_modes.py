from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["synthetic", "literature_calibrated"]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_mode_outputs(mode: str) -> dict[str, pd.DataFrame]:
    return {
        "scenario_suite_summary": safe_read_csv(RESULTS_DIR / f"scenario_suite_summary_{mode}.csv"),
        "hospital_bottleneck_summary": safe_read_csv(RESULTS_DIR / f"hospital_bottleneck_summary_{mode}.csv"),
        "hospital_bottleneck_decomposition": safe_read_csv(RESULTS_DIR / f"hospital_bottleneck_decomposition_{mode}.csv"),
        "hospital_time_series_summary": safe_read_csv(RESULTS_DIR / f"hospital_time_series_summary_{mode}.csv"),
    }


def build_policy_kpi_mode_comparison(all_mode_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    pieces = []

    for mode, data in all_mode_data.items():
        df = data["scenario_suite_summary"].copy()
        df["mode"] = mode
        pieces.append(df)

    stacked = pd.concat(pieces, ignore_index=True)

    value_cols = [
        "total_unsafe_excess_mean",
        "total_overflow_excess_mean",
        "total_surge_gap_mean",
        "max_utilization_ratio_mean",
        "num_unsafe_rows_mean",
        "total_blocked_arrivals_mean",
    ]
    value_cols = [c for c in value_cols if c in stacked.columns]

    pivoted = (
        stacked.pivot_table(
            index=["scenario", "policy_name"],
            columns="mode",
            values=value_cols,
            aggfunc="first",
        )
    )

    pivoted.columns = [
        f"{metric}_{mode}"
        for metric, mode in pivoted.columns
    ]
    pivoted = pivoted.reset_index()

    for metric in value_cols:
        syn_col = f"{metric}_synthetic"
        lit_col = f"{metric}_literature_calibrated"
        if syn_col in pivoted.columns and lit_col in pivoted.columns:
            pivoted[f"{metric}_abs_change_lit_minus_syn"] = (
                pivoted[lit_col] - pivoted[syn_col]
            )
            denom = pivoted[syn_col].replace(0, pd.NA)
            pivoted[f"{metric}_pct_change_lit_vs_syn"] = (
                (pivoted[lit_col] - pivoted[syn_col]) / denom * 100.0
            )

    return pivoted.sort_values(["scenario", "policy_name"]).reset_index(drop=True)


def build_best_policy_comparison(all_mode_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    for mode, data in all_mode_data.items():
        df = data["scenario_suite_summary"].copy()
        if df.empty:
            continue

        for scenario, sdf in df.groupby("scenario"):
            sdf = sdf.copy()

            best_unsafe = sdf.sort_values("total_unsafe_excess_mean").iloc[0]
            best_util = sdf.sort_values("max_utilization_ratio_mean").iloc[0]
            best_blocked = sdf.sort_values("total_blocked_arrivals_mean").iloc[0]

            rows.append(
                {
                    "mode": mode,
                    "scenario": scenario,
                    "best_policy_unsafe_excess": best_unsafe["policy_name"],
                    "best_unsafe_excess_value": best_unsafe["total_unsafe_excess_mean"],
                    "best_policy_max_utilization": best_util["policy_name"],
                    "best_max_utilization_value": best_util["max_utilization_ratio_mean"],
                    "best_policy_blocked_arrivals": best_blocked["policy_name"],
                    "best_blocked_arrivals_value": best_blocked["total_blocked_arrivals_mean"],
                }
            )

    return pd.DataFrame(rows).sort_values(["scenario", "mode"]).reset_index(drop=True)


def build_hospital_bottleneck_mode_comparison(all_mode_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    pieces = []

    for mode, data in all_mode_data.items():
        df = data["hospital_bottleneck_summary"].copy()
        df["mode"] = mode
        pieces.append(df)

    stacked = pd.concat(pieces, ignore_index=True)

    value_cols = [
        "unsafe_excess_mean",
        "overflow_excess_mean",
        "max_utilization_mean",
        "num_unsafe_rows_mean",
        "transfer_activity_mean",
        "bottleneck_score_mean",
    ]
    value_cols = [c for c in value_cols if c in stacked.columns]

    pivoted = (
        stacked.pivot_table(
            index=["scenario", "policy_name", "hospital_id"],
            columns="mode",
            values=value_cols,
            aggfunc="first",
        )
    )

    pivoted.columns = [
        f"{metric}_{mode}"
        for metric, mode in pivoted.columns
    ]
    pivoted = pivoted.reset_index()

    for metric in value_cols:
        syn_col = f"{metric}_synthetic"
        lit_col = f"{metric}_literature_calibrated"
        if syn_col in pivoted.columns and lit_col in pivoted.columns:
            pivoted[f"{metric}_abs_change_lit_minus_syn"] = (
                pivoted[lit_col] - pivoted[syn_col]
            )

    return pivoted.sort_values(["scenario", "policy_name", "hospital_id"]).reset_index(drop=True)


def build_decomposition_mode_comparison(all_mode_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    pieces = []

    for mode, data in all_mode_data.items():
        df = data["hospital_bottleneck_decomposition"].copy()
        df["mode"] = mode
        pieces.append(df)

    stacked = pd.concat(pieces, ignore_index=True)

    value_cols = [
        "util_component_mean",
        "unsafe_component_mean",
        "overflow_component_mean",
        "event_component_mean",
        "transfer_component_mean",
        "bottleneck_score_mean",
    ]
    value_cols = [c for c in value_cols if c in stacked.columns]

    pivoted = (
        stacked.pivot_table(
            index=["scenario", "policy_name", "hospital_id"],
            columns="mode",
            values=value_cols,
            aggfunc="first",
        )
    )

    pivoted.columns = [
        f"{metric}_{mode}"
        for metric, mode in pivoted.columns
    ]
    pivoted = pivoted.reset_index()

    for metric in value_cols:
        syn_col = f"{metric}_synthetic"
        lit_col = f"{metric}_literature_calibrated"
        if syn_col in pivoted.columns and lit_col in pivoted.columns:
            pivoted[f"{metric}_abs_change_lit_minus_syn"] = (
                pivoted[lit_col] - pivoted[syn_col]
            )

    return pivoted.sort_values(["scenario", "policy_name", "hospital_id"]).reset_index(drop=True)


def build_temporal_mode_comparison(all_mode_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    pieces = []

    for mode, data in all_mode_data.items():
        df = data["hospital_time_series_summary"].copy()
        df["mode"] = mode
        pieces.append(df)

    stacked = pd.concat(pieces, ignore_index=True)

    value_cols = [
        "mean_daily_bottleneck_score",
        "max_daily_bottleneck_score",
        "mean_daily_unsafe_excess",
        "max_daily_unsafe_excess",
        "mean_daily_overflow_excess",
        "max_daily_overflow_excess",
        "mean_daily_utilization",
        "max_daily_utilization",
        "days_above_safe_utilization",
        "peak_bottleneck_day",
        "peak_bottleneck_score",
    ]
    value_cols = [c for c in value_cols if c in stacked.columns]

    pivoted = (
        stacked.pivot_table(
            index=["scenario", "policy_name", "hospital_id"],
            columns="mode",
            values=value_cols,
            aggfunc="first",
        )
    )

    pivoted.columns = [
        f"{metric}_{mode}"
        for metric, mode in pivoted.columns
    ]
    pivoted = pivoted.reset_index()

    for metric in value_cols:
        syn_col = f"{metric}_synthetic"
        lit_col = f"{metric}_literature_calibrated"
        if syn_col in pivoted.columns and lit_col in pivoted.columns:
            pivoted[f"{metric}_abs_change_lit_minus_syn"] = (
                pivoted[lit_col] - pivoted[syn_col]
            )

    return pivoted.sort_values(["scenario", "policy_name", "hospital_id"]).reset_index(drop=True)


def build_top_hospital_shift_table(all_mode_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    for mode, data in all_mode_data.items():
        df = data["hospital_bottleneck_summary"].copy()
        if df.empty:
            continue

        for (scenario, policy_name), sdf in df.groupby(["scenario", "policy_name"]):
            top_row = sdf.sort_values("bottleneck_score_mean", ascending=False).iloc[0]
            rows.append(
                {
                    "mode": mode,
                    "scenario": scenario,
                    "policy_name": policy_name,
                    "top_hospital": top_row["hospital_id"],
                    "top_bottleneck_score": top_row["bottleneck_score_mean"],
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["scenario", "policy_name", "mode"]).reset_index(drop=True)


def build_executive_delta_table(
    policy_kpi_comp: pd.DataFrame,
    top_hospital_shift: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in policy_kpi_comp.iterrows():
        scenario = row["scenario"]
        policy_name = row["policy_name"]

        top_rows = top_hospital_shift[
            (top_hospital_shift["scenario"] == scenario)
            & (top_hospital_shift["policy_name"] == policy_name)
        ].copy()

        syn_top = None
        lit_top = None

        if not top_rows.empty:
            syn_rows = top_rows[top_rows["mode"] == "synthetic"]
            lit_rows = top_rows[top_rows["mode"] == "literature_calibrated"]
            syn_top = syn_rows.iloc[0]["top_hospital"] if not syn_rows.empty else None
            lit_top = lit_rows.iloc[0]["top_hospital"] if not lit_rows.empty else None

        rows.append(
            {
                "scenario": scenario,
                "policy_name": policy_name,
                "unsafe_excess_change": row.get("total_unsafe_excess_mean_abs_change_lit_minus_syn"),
                "utilization_change": row.get("max_utilization_ratio_mean_abs_change_lit_minus_syn"),
                "blocked_arrivals_change": row.get("total_blocked_arrivals_mean_abs_change_lit_minus_syn"),
                "top_hospital_synthetic": syn_top,
                "top_hospital_literature": lit_top,
                "top_hospital_changed": syn_top != lit_top,
            }
        )

    return pd.DataFrame(rows).sort_values(["scenario", "policy_name"]).reset_index(drop=True)


def main() -> None:
    all_mode_data = {mode: load_mode_outputs(mode) for mode in MODES}

    policy_kpi_comp = build_policy_kpi_mode_comparison(all_mode_data)
    best_policy_comp = build_best_policy_comparison(all_mode_data)
    hospital_bottleneck_comp = build_hospital_bottleneck_mode_comparison(all_mode_data)
    decomp_comp = build_decomposition_mode_comparison(all_mode_data)
    temporal_comp = build_temporal_mode_comparison(all_mode_data)
    top_hospital_shift = build_top_hospital_shift_table(all_mode_data)
    executive_delta = build_executive_delta_table(policy_kpi_comp, top_hospital_shift)

    outputs = {
        "compare_modes_policy_kpis.csv": policy_kpi_comp,
        "compare_modes_best_policies.csv": best_policy_comp,
        "compare_modes_hospital_bottlenecks.csv": hospital_bottleneck_comp,
        "compare_modes_bottleneck_decomposition.csv": decomp_comp,
        "compare_modes_temporal_summary.csv": temporal_comp,
        "compare_modes_top_hospital_shift.csv": top_hospital_shift,
        "compare_modes_executive_delta.csv": executive_delta,
    }

    for filename, df in outputs.items():
        path = TABLES_DIR / filename
        df.to_csv(path, index=False)

    print("\nSaved comparison tables:")
    for filename in outputs:
        print(TABLES_DIR / filename)

    print("\n=== Best policy comparison ===")
    print(best_policy_comp)

    print("\n=== Executive delta summary ===")
    print(executive_delta)


if __name__ == "__main__":
    main()