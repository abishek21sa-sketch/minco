"""Mathematical monotonicity checks for the capacity optimization model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.dataclasses import HealthcareInstance
from src.config.loader import load_healthcare_instance
from src.config.paths import DATA_DIR, RESULTS_DIR
from src.validation.optimization_checks import assert_optimal_model
from src.validation.run_manifest import new_run_id, write_run_manifest


def _clone_instance(
    instance: HealthcareInstance,
    *,
    capacities: pd.DataFrame | None = None,
    arrivals: pd.DataFrame | None = None,
    transfer_lanes: pd.DataFrame | None = None,
) -> HealthcareInstance:
    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=type(instance.capacities)(df=(capacities if capacities is not None else instance.capacities.df).copy()),
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=type(instance.transfer_lanes)(
            df=(transfer_lanes if transfer_lanes is not None else instance.transfer_lanes.df).copy()
        ),
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=type(instance.arrivals)(df=(arrivals if arrivals is not None else instance.arrivals.df).copy()),
        transitions=instance.transitions,
    )


def _solve_objective(instance: HealthcareInstance, name: str) -> dict[str, float]:
    from src.optimizer.model_builder import solve_network_capacity_model

    model, *_ = solve_network_capacity_model(instance, model_name=name, output_flag=0)
    return assert_optimal_model(model)


def run_optimization_sensitivity_validation(
    *,
    capacity_scale: float = 1.10,
    demand_scale: float = 1.10,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify feasible-set monotonicity under capacity and demand perturbations."""
    if capacity_scale < 1.0 or demand_scale < 1.0:
        raise ValueError("Validation scales must be at least 1.0")
    output_dir = Path(output_dir or (RESULTS_DIR / "validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    instance = load_healthcare_instance(DATA_DIR)

    capacity_df = instance.capacities.df.copy()
    capacity_df["base_capacity"] *= capacity_scale
    demand_df = instance.arrivals.df.copy()
    demand_df["arrivals"] *= demand_scale
    transfers_disabled = instance.transfer_lanes.df.copy()
    transfers_disabled["allowed"] = 0

    baseline = _solve_objective(instance, "wp2_baseline")
    capacity_relaxed = _solve_objective(
        _clone_instance(instance, capacities=capacity_df), "wp2_capacity_relaxed"
    )
    demand_stressed = _solve_objective(
        _clone_instance(instance, arrivals=demand_df), "wp2_demand_stressed"
    )
    no_transfers = _solve_objective(
        _clone_instance(instance, transfer_lanes=transfers_disabled), "wp2_no_transfers"
    )

    tolerance = 1e-6
    checks = {
        "capacity_relaxation_does_not_worsen_objective": (
            capacity_relaxed["objective_value"] <= baseline["objective_value"] + tolerance
        ),
        "demand_stress_does_not_improve_objective": (
            demand_stressed["objective_value"] >= baseline["objective_value"] - tolerance
        ),
        "transfer_restriction_does_not_improve_objective": (
            no_transfers["objective_value"] >= baseline["objective_value"] - tolerance
        ),
    }
    rows = []
    for scenario, diagnostics in {
        "baseline": baseline,
        "capacity_relaxed": capacity_relaxed,
        "demand_stressed": demand_stressed,
        "transfers_disabled": no_transfers,
    }.items():
        rows.append({"scenario": scenario, **diagnostics})
    output_path = output_dir / "optimization_sensitivity_validation.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    status = "passed" if all(checks.values()) else "failed"
    run_id = new_run_id("optimization_sensitivity")
    payload = {
        "run_id": run_id,
        "status": status,
        "capacity_scale": capacity_scale,
        "demand_scale": demand_scale,
        "checks": checks,
        "objectives": {row["scenario"]: row["objective_value"] for row in rows},
        "output_path": str(output_path),
    }
    manifest_path, manifest_sha = write_run_manifest(
        run_id=run_id,
        run_type="optimization_sensitivity_validation",
        parameters={"capacity_scale": capacity_scale, "demand_scale": demand_scale},
        input_paths=[DATA_DIR / "base_instance", DATA_DIR / "transitions"],
        output_paths=[output_path],
        metrics=payload["objectives"],
        model_info={"model_class": "continuous_linear_program", "solver": "Gurobi"},
        notes=["Checks are feasible-set monotonicity tests, not real-world outcome validation."],
    )
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = manifest_sha
    if status != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Optimization sensitivity failed checks: {failed}")
    return payload
