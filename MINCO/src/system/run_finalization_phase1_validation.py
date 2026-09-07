"""Finalization Phase 1 evidence gate: AI + IE + OR integration foundation."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ai.demand_forecast_quantile import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_PATH,
    DEFAULT_VALIDATION_PATH,
    forecast_next_day,
    train_validate_demand_forecast,
)
from src.config.loader import load_healthcare_instance
from src.config.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from src.contracts.common import ExecutionContext
from src.contracts.surge_plan import SurgePlanRequest
from src.data_engineering.synthetic_arrival_history import (
    DEFAULT_HISTORY_DAYS,
    DEFAULT_HISTORY_SEED,
    generate_synthetic_arrival_history,
)
from src.ie.healthcare_flow import (
    build_resource_pressure_projection,
    expected_resource_days_by_cohort,
)
from src.services.container import build_service_container
from dashboard.live_sandbox import clone_instance_with_whatif
from src.validation.run_manifest import RUN_MANIFEST_DIR, new_run_id, write_run_manifest
from src.validation.solver_readiness import check_gurobi_readiness
from src.version import __version__

HISTORY_PATH = RESULTS_DIR / "reference_history" / "synthetic_arrival_history.csv"
RESOURCE_DAYS_PATH = RESULTS_DIR / "validation" / "ie_expected_resource_days.csv"
P50_PRESSURE_PATH = RESULTS_DIR / "validation" / "ie_next_day_pressure_p50.csv"
P95_PRESSURE_PATH = RESULTS_DIR / "validation" / "ie_next_day_pressure_p95.csv"
REPORT_PATH = RESULTS_DIR / "validation" / "finalization_phase1_validation_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def run_validation(*, allow_solver_skip: bool | None = None) -> dict[str, Any]:
    if allow_solver_skip is None:
        allow_solver_skip = os.getenv("MINCO_ALLOW_SOLVER_SKIP", "0") == "1"

    instance = load_healthcare_instance(DATA_DIR)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.joinpath("validation").mkdir(parents=True, exist_ok=True)

    history = generate_synthetic_arrival_history(
        instance,
        n_days=DEFAULT_HISTORY_DAYS,
        seed=DEFAULT_HISTORY_SEED,
    )
    history.to_csv(HISTORY_PATH, index=False)

    forecast_validation = train_validate_demand_forecast(
        history,
        model_path=DEFAULT_MODEL_PATH,
        validation_path=DEFAULT_VALIDATION_PATH,
        prediction_path=DEFAULT_PREDICTION_PATH,
    )
    next_day = forecast_next_day(history, model_path=DEFAULT_MODEL_PATH)

    resource_days = expected_resource_days_by_cohort(instance)
    resource_days.to_csv(RESOURCE_DAYS_PATH, index=False)
    p50 = build_resource_pressure_projection(
        instance,
        next_day,
        quantile_column="predicted_p50",
    )
    p95 = build_resource_pressure_projection(
        instance,
        next_day,
        quantile_column="predicted_p95",
    )
    p50.to_csv(P50_PRESSURE_PATH, index=False)
    p95.to_csv(P95_PRESSURE_PATH, index=False)

    transfer_lanes = instance.transfer_lanes.df.copy()
    transfer_degraded = clone_instance_with_whatif(
        instance, transfer_capacity_multiplier=0.80
    ).transfer_lanes.df.copy()
    transfer_topology_preserved = transfer_degraded["allowed"].tolist() == transfer_lanes["allowed"].tolist()
    transfer_capacity_scaled = bool(
        (
            transfer_degraded["transfer_capacity"].astype(float).to_numpy()
            == (transfer_lanes["transfer_capacity"].astype(float) * 0.80).to_numpy()
        ).all()
    )

    solver = check_gurobi_readiness().to_dict()
    integrated_plan: dict[str, Any] | None = None
    integrated_check = False
    transfer_solver_check = False
    if solver["license_verified"]:
        services = build_service_container()
        response = services.surge_planning.plan(
            SurgePlanRequest(
                scenario_id="baseline",
                n_replications=3,
                context=ExecutionContext(source="finalization_phase1_gate"),
            )
        )
        integrated_plan = {
            "run_id": response.run_id,
            "decision": response.decision.model_dump(mode="json"),
            "forecast_summary": response.forecast_summary.model_dump(mode="json"),
            "optimized_actions": response.optimized_actions.model_dump(mode="json"),
            "scenario_metrics": response.scenario_evaluation.metrics.model_dump(mode="json"),
            "manifest": (
                None if response.metadata.evidence is None else response.metadata.evidence.model_dump()
            ),
        }
        integrated_check = response.scenario_evaluation.metrics.solver_status_label == "OPTIMAL"

        from src.baselines.policy_baselines import build_optimized_network_policy_snapshot
        from src.validation.optimization_checks import validate_network_solution_tables

        snapshot = build_optimized_network_policy_snapshot(instance)
        transfer_checks = validate_network_solution_tables(snapshot, instance)
        transfer_solver_check = transfer_checks["transfer_capacity_max_excess"] <= 1e-6
        integrated_plan["transfer_capacity_validation"] = transfer_checks
    elif allow_solver_skip:
        integrated_plan = {
            "status": "skipped",
            "reason": "Gurobi is unavailable in this build environment; laptop acceptance must run the full gate.",
        }
        integrated_check = True
        transfer_solver_check = True

    checks = {
        "synthetic_history_complete": len(history) == DEFAULT_HISTORY_DAYS * 3 * 4,
        "forecast_validation_passed": forecast_validation["status"] == "passed",
        "forecast_beats_seasonal_baseline": forecast_validation["metrics"][
            "mae_improvement_vs_seasonal_naive"
        ]
        >= 0.10,
        "forecast_interval_calibrated": forecast_validation["metrics"][
            "prediction_interval_90_coverage"
        ]
        >= 0.85,
        "ie_resource_days_positive": bool(
            (resource_days["expected_resource_days_per_arrival"] > 0).all()
        ),
        "ie_projection_complete": len(p50) == 6 and len(p95) == 6,
        "transfer_topology_preserved_under_capacity_scaling": transfer_topology_preserved,
        "transfer_capacity_scaled_quantitatively": transfer_capacity_scaled,
        "solver_transfer_capacity_constraints_verified_or_build_skip": transfer_solver_check,
        "solver_or_allowed_build_skip": bool(solver["license_verified"] or allow_solver_skip),
        "integrated_ai_ie_or_simulation_path": integrated_check,
    }
    status = "passed" if all(checks.values()) else "failed"
    run_id = new_run_id("finalization_phase1")

    manifest_path, manifest_hash = write_run_manifest(
        run_id=run_id,
        run_type="finalization_phase1_validation",
        parameters={
            "history_days": DEFAULT_HISTORY_DAYS,
            "history_seed": DEFAULT_HISTORY_SEED,
            "allow_solver_skip": allow_solver_skip,
        },
        input_paths=[
            DATA_DIR / "base_instance",
            DATA_DIR / "transitions",
            HISTORY_PATH,
        ],
        metrics={
            "status": status,
            "checks": checks,
            "forecast_metrics": forecast_validation["metrics"],
            "max_p95_projected_utilization": float(p95["projected_utilization"].max()),
            "p95_rows_above_safe_capacity": int(p95["above_safe_capacity"].sum()),
            "solver_readiness": solver,
        },
        model_info={
            "ai": "quantile_gradient_boosting_next_day_arrivals",
            "industrial_engineering": "absorbing_markov_fundamental_matrix_plus_littles_law",
            "optimization": "gurobi_continuous_network_capacity_lp",
            "simulation": "seeded_stochastic_state_transition_twin",
        },
        notes=[
            "The demand forecast is validated on synthetic historical replay, not real hospital data.",
            "Earlier replication-index forecasting artifacts are retained as research history but are not treated as temporal deployment evidence.",
            "The IE projection is a steady-state capacity-planning approximation.",
        ],
        manifest_dir=RUN_MANIFEST_DIR,
    )

    report = {
        "work_package": "finalization_phase1_decision_engine_closure",
        "version": __version__,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "status": status,
        "checks": checks,
        "solver_readiness": solver,
        "synthetic_history": {
            "path": _relative(HISTORY_PATH),
            "rows": len(history),
            "seed": DEFAULT_HISTORY_SEED,
            "evidence_label": "SYNTHETIC",
        },
        "demand_forecast_validation": forecast_validation,
        "transfer_capacity_model": {
            "topology_field": "allowed",
            "capacity_field": "transfer_capacity",
            "synthetic_capacity_basis": "20_percent_of_source_nominal_ICU_capacity_per_lane_day",
            "total_base_transfer_capacity_per_day": float(
                transfer_lanes.loc[transfer_lanes["allowed"] == 1, "transfer_capacity"].sum()
            ),
        },
        "industrial_engineering": {
            "expected_resource_days_path": _relative(RESOURCE_DAYS_PATH),
            "p50_pressure_path": _relative(P50_PRESSURE_PATH),
            "p95_pressure_path": _relative(P95_PRESSURE_PATH),
            "max_p95_projected_utilization": float(p95["projected_utilization"].max()),
            "p95_rows_above_safe_capacity": int(p95["above_safe_capacity"].sum()),
        },
        "integrated_surge_plan": integrated_plan,
        "claim_boundary": (
            "Phase 1 establishes a computational AI + IE + OR + simulation workflow on the "
            "bundled synthetic reference case. Real-hospital calibration and deployment validation remain pending."
        ),
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": manifest_hash,
        "required_follow_up": (
            [] if solver["license_verified"] else ["Run the full gate on the licensed Windows laptop."]
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_validation()
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
