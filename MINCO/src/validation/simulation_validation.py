"""Verification and statistical validation for the stochastic state-transition twin."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.loader import load_healthcare_instance
from src.config.paths import DATA_DIR, RESULTS_DIR
from src.markov.forecast_engine import run_forecast
from src.simulation.twin_engine import simulate_stochastic_twin
from src.validation.run_manifest import new_run_id, write_run_manifest


STATE_KEYS = ["day", "hospital_id", "cohort", "state"]
ARRIVAL_KEYS = ["day", "hospital_id", "cohort"]


@dataclass(frozen=True)
class SimulationValidationThresholds:
    maximum_mass_error: float = 0.0
    maximum_state_mae: float = 0.25
    maximum_state_normalized_mae: float = 0.05
    minimum_expectation_coverage_95: float = 0.90
    maximum_arrival_mean_relative_error: float = 0.05
    minimum_arrival_variance_ratio_median: float = 0.80
    maximum_arrival_variance_ratio_median: float = 1.20


def validate_patient_mass_conservation(
    state_trajectories: pd.DataFrame,
    realized_arrivals: pd.DataFrame,
) -> dict[str, float]:
    """Verify cumulative patient mass for every day, hospital, and cohort.

    Discharged and Dead are absorbing states in the bundled transition matrices,
    so the complete state vector must equal cumulative arrivals when the initial
    state is zero.
    """
    state_totals = (
        state_trajectories.groupby(["day", "hospital_id", "cohort"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "state_total"})
    )
    arrivals = realized_arrivals.sort_values(ARRIVAL_KEYS).copy()
    arrivals["cumulative_arrivals"] = arrivals.groupby(
        ["hospital_id", "cohort"], sort=False
    )["realized_arrivals"].cumsum()
    joined = state_totals.merge(
        arrivals[ARRIVAL_KEYS + ["cumulative_arrivals"]],
        on=ARRIVAL_KEYS,
        how="left",
        validate="one_to_one",
    )
    if joined["cumulative_arrivals"].isna().any():
        raise AssertionError("State rows could not be matched to realized arrivals")
    errors = (joined["state_total"] - joined["cumulative_arrivals"]).abs()
    return {
        "mass_conservation_rows": float(len(joined)),
        "mass_conservation_max_error": float(errors.max() if len(errors) else 0.0),
        "mass_conservation_total_error": float(errors.sum()),
    }


def _expectation_convergence_table(
    expected: pd.DataFrame,
    realized: pd.DataFrame,
) -> pd.DataFrame:
    stats = (
        realized.groupby(STATE_KEYS, as_index=False)["count"]
        .agg(realized_mean="mean", realized_std="std", replications="count")
    )
    table = expected.merge(stats, on=STATE_KEYS, how="inner", validate="one_to_one")
    table["realized_std"] = table["realized_std"].fillna(0.0)
    table["standard_error"] = table["realized_std"] / np.sqrt(table["replications"])
    table["ci95_lower"] = table["realized_mean"] - 1.96 * table["standard_error"]
    table["ci95_upper"] = table["realized_mean"] + 1.96 * table["standard_error"]
    table["absolute_error"] = (table["realized_mean"] - table["expected_count"]).abs()
    table["normalized_absolute_error"] = table["absolute_error"] / (
        table["expected_count"].abs() + 1.0
    )
    table["expected_in_ci95"] = (
        (table["expected_count"] >= table["ci95_lower"])
        & (table["expected_count"] <= table["ci95_upper"])
    )
    return table.sort_values(STATE_KEYS).reset_index(drop=True)


def _arrival_calibration_table(realized_arrivals: pd.DataFrame) -> pd.DataFrame:
    table = (
        realized_arrivals.groupby(ARRIVAL_KEYS, as_index=False)
        .agg(
            poisson_lambda=("arrivals", "first"),
            realized_mean=("realized_arrivals", "mean"),
            realized_variance=("realized_arrivals", "var"),
            replications=("realized_arrivals", "count"),
        )
        .sort_values(ARRIVAL_KEYS)
        .reset_index(drop=True)
    )
    table["realized_variance"] = table["realized_variance"].fillna(0.0)
    table["mean_absolute_error"] = (
        table["realized_mean"] - table["poisson_lambda"]
    ).abs()
    table["variance_to_mean_ratio"] = np.where(
        table["poisson_lambda"] > 0,
        table["realized_variance"] / table["poisson_lambda"],
        np.nan,
    )
    return table


def run_simulation_validation(
    *,
    n_replications: int = 200,
    base_seed: int = 20260730,
    thresholds: SimulationValidationThresholds | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run seeded Monte Carlo verification against deterministic expectations."""
    if n_replications < 30:
        raise ValueError("Simulation validation requires at least 30 replications")
    thresholds = thresholds or SimulationValidationThresholds()
    output_dir = Path(output_dir or (RESULTS_DIR / "validation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    instance = load_healthcare_instance(DATA_DIR)
    expected = run_forecast(instance)["state_trajectories"]

    state_frames: list[pd.DataFrame] = []
    arrival_frames: list[pd.DataFrame] = []
    mass_records: list[dict[str, float]] = []

    for replication in range(n_replications):
        seed = base_seed + replication
        outputs = simulate_stochastic_twin(instance, rng=np.random.default_rng(seed))
        mass = validate_patient_mass_conservation(
            outputs["state_trajectories"], outputs["realized_arrivals"]
        )
        mass["replication"] = float(replication + 1)
        mass_records.append(mass)

        states = outputs["state_trajectories"].copy()
        states["replication"] = replication + 1
        state_frames.append(states)

        arrivals = outputs["realized_arrivals"].copy()
        arrivals["replication"] = replication + 1
        arrival_frames.append(arrivals)

    all_states = pd.concat(state_frames, ignore_index=True)
    all_arrivals = pd.concat(arrival_frames, ignore_index=True)
    convergence = _expectation_convergence_table(expected, all_states)
    arrival_calibration = _arrival_calibration_table(all_arrivals)
    mass_table = pd.DataFrame(mass_records)

    convergence_path = output_dir / "simulation_expectation_convergence.csv"
    arrivals_path = output_dir / "arrival_process_calibration.csv"
    mass_path = output_dir / "simulation_mass_conservation.csv"
    summary_path = output_dir / "simulation_validation_summary.json"
    convergence.to_csv(convergence_path, index=False)
    arrival_calibration.to_csv(arrivals_path, index=False)
    mass_table.to_csv(mass_path, index=False)

    weighted_arrival_error = float(
        arrival_calibration["mean_absolute_error"].sum()
        / max(arrival_calibration["poisson_lambda"].sum(), 1.0)
    )
    variance_ratios = arrival_calibration["variance_to_mean_ratio"].dropna()
    metrics = {
        "n_replications": int(n_replications),
        "mass_conservation_max_error": float(
            mass_table["mass_conservation_max_error"].max()
        ),
        "state_expectation_mae": float(convergence["absolute_error"].mean()),
        "state_expectation_rmse": float(
            np.sqrt(np.mean(np.square(convergence["absolute_error"])))
        ),
        "state_expectation_normalized_mae": float(
            convergence["normalized_absolute_error"].mean()
        ),
        "state_expectation_coverage_95": float(
            convergence["expected_in_ci95"].mean()
        ),
        "arrival_mean_weighted_relative_error": weighted_arrival_error,
        "arrival_variance_ratio_median": float(variance_ratios.median()),
        "arrival_variance_ratio_mean": float(variance_ratios.mean()),
    }
    checks = {
        "mass_conservation": metrics["mass_conservation_max_error"]
        <= thresholds.maximum_mass_error,
        "state_mean_accuracy": metrics["state_expectation_mae"]
        <= thresholds.maximum_state_mae,
        "state_normalized_accuracy": metrics["state_expectation_normalized_mae"]
        <= thresholds.maximum_state_normalized_mae,
        "expectation_coverage": metrics["state_expectation_coverage_95"]
        >= thresholds.minimum_expectation_coverage_95,
        "arrival_mean_calibration": metrics["arrival_mean_weighted_relative_error"]
        <= thresholds.maximum_arrival_mean_relative_error,
        "arrival_variance_calibration": (
            thresholds.minimum_arrival_variance_ratio_median
            <= metrics["arrival_variance_ratio_median"]
            <= thresholds.maximum_arrival_variance_ratio_median
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    run_id = new_run_id("simulation_validation")
    payload = {
        "run_id": run_id,
        "status": status,
        "method": "seeded_monte_carlo_vs_markov_expectation",
        "model_classification": "stochastic_state_transition_twin",
        "not_validated_as": "discrete_event_or_real_time_hospital_digital_twin",
        "parameters": {"n_replications": n_replications, "base_seed": base_seed},
        "thresholds": asdict(thresholds),
        "metrics": metrics,
        "checks": checks,
        "outputs": {
            "expectation_convergence": str(convergence_path),
            "arrival_calibration": str(arrivals_path),
            "mass_conservation": str(mass_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest_path, manifest_sha = write_run_manifest(
        run_id=run_id,
        run_type="simulation_validation",
        parameters=payload["parameters"],
        input_paths=[DATA_DIR / "base_instance", DATA_DIR / "transitions"],
        output_paths=[convergence_path, arrivals_path, mass_path],
        metrics=metrics,
        model_info={
            "classification": "stochastic_state_transition_twin",
            "formal_des": False,
        },
        notes=[
            "Validation is against the synthetic reference model's known Markov and Poisson assumptions.",
            "This does not establish external validity for a real hospital system.",
        ],
    )
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = manifest_sha
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if status != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Simulation validation failed checks: {failed}")
    return payload
