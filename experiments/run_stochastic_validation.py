from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.loader import load_healthcare_instance
from src.simulation.policy_runner import run_multiple_stochastic_replications


def summarize_replication_table(
    replications_df: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """
    Build a one-row summary table from replication KPIs.
    """
    summary = {
        "scenario": scenario_name,
        "n_replications": int(len(replications_df)),
    }

    metric_cols = [
        "total_unsafe_excess",
        "total_overflow_excess",
        "total_surge_gap",
        "max_utilization_ratio",
        "max_safe_utilization_ratio",
        "num_unsafe_rows",
        "num_overflow_rows",
        "num_surge_gap_rows",
    ]

    for col in metric_cols:
        summary[f"{col}_mean"] = float(replications_df[col].mean())
        summary[f"{col}_std"] = float(replications_df[col].std(ddof=1)) if len(replications_df) > 1 else 0.0
        summary[f"{col}_min"] = float(replications_df[col].min())
        summary[f"{col}_max"] = float(replications_df[col].max())

    return pd.DataFrame([summary])


def run_baseline_validation(
    n_replications: int = 50,
    base_seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run baseline stochastic validation.
    """
    instance = load_healthcare_instance()

    replications = run_multiple_stochastic_replications(
        instance=instance,
        n_replications=n_replications,
        base_seed=base_seed,
    )

    summary = summarize_replication_table(
        replications_df=replications,
        scenario_name="baseline",
    )

    return replications, summary


def run_h3_c4_shock_validation(
    n_replications: int = 50,
    base_seed: int = 123,
    shock_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a shock scenario focused on H3 critical-care arrivals
    during the late-horizon congestion window.
    """
    instance = load_healthcare_instance()

    replications = run_multiple_stochastic_replications(
        instance=instance,
        n_replications=n_replications,
        base_seed=base_seed,
        arrival_shock_multiplier=shock_multiplier,
        shock_cohort="c4",
        shock_hospital_id="H3",
        shock_start_day=5,
        shock_end_day=7,
    )

    summary = summarize_replication_table(
        replications_df=replications,
        scenario_name="h3_c4_shock",
    )

    summary["shock_multiplier"] = float(shock_multiplier)

    return replications, summary


def save_validation_outputs(
    replications_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    replications_path: Path,
    summary_path: Path,
) -> None:
    """
    Save replication table and summary table to CSV.
    """
    replications_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    replications_df.to_csv(replications_path, index=False)
    summary_df.to_csv(summary_path, index=False)


def main() -> None:
    """
    Run baseline and shocked stochastic validation studies.
    """
    output_dir = Path("results")

    baseline_replications, baseline_summary = run_baseline_validation(
        n_replications=50,
        base_seed=123,
    )

    shock_replications, shock_summary = run_h3_c4_shock_validation(
        n_replications=50,
        base_seed=123,
        shock_multiplier=1.5,
    )

    save_validation_outputs(
        replications_df=baseline_replications,
        summary_df=baseline_summary,
        replications_path=output_dir / "stochastic_validation_baseline_replications.csv",
        summary_path=output_dir / "stochastic_validation_baseline_summary.csv",
    )

    save_validation_outputs(
        replications_df=shock_replications,
        summary_df=shock_summary,
        replications_path=output_dir / "stochastic_validation_h3_c4_shock_replications.csv",
        summary_path=output_dir / "stochastic_validation_h3_c4_shock_summary.csv",
    )

    combined_summary = pd.concat(
        [baseline_summary, shock_summary],
        ignore_index=True,
    )
    combined_summary_path = output_dir / "stochastic_validation_combined_summary.csv"
    combined_summary.to_csv(combined_summary_path, index=False)

    print("\nBASELINE SUMMARY")
    print(baseline_summary)

    print("\nH3 C4 SHOCK SUMMARY")
    print(shock_summary)

    print("\nCOMBINED SUMMARY")
    print(combined_summary)

    print("\nSaved files:")
    print(output_dir / "stochastic_validation_baseline_replications.csv")
    print(output_dir / "stochastic_validation_baseline_summary.csv")
    print(output_dir / "stochastic_validation_h3_c4_shock_replications.csv")
    print(output_dir / "stochastic_validation_h3_c4_shock_summary.csv")
    print(combined_summary_path)


if __name__ == "__main__":
    main()