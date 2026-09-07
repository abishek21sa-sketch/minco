"""Internal benchmark readiness and external-calibration boundary for MINCO."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


from src.config.loader import load_healthcare_instance
from src.config.paths import DATA_DIR, RESULTS_DIR
from src.validation.run_manifest import new_run_id, write_run_manifest

REPORT_NAME = "benchmark_calibration_readiness.json"


def run_benchmark_calibration_readiness(
    *,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    manifest_dir: Path | None = None,
) -> dict[str, object]:
    """Validate the synthetic reference benchmark and state the external-data boundary.

    This does not claim calibration to a real hospital. It verifies that the bundled
    reference case is internally coherent and defines the event fields required for
    future historical or public-benchmark calibration.
    """
    instance = load_healthcare_instance(data_dir)
    arrivals = instance.arrivals.df
    hospitals = instance.hospitals.df
    capacities = instance.capacities.df
    safe = instance.safe_thresholds.df
    lanes = instance.transfer_lanes.df
    elective = instance.elective_bounds.df

    cohorts = sorted(arrivals["cohort"].astype(str).unique().tolist())
    days = sorted(arrivals["day"].unique().tolist())
    expected_arrival_rows = len(hospitals) * len(cohorts) * len(days)
    elective_cohorts = sorted(elective["cohort"].astype(str).unique().tolist())
    expected_elective_rows = len(hospitals) * len(elective_cohorts) * len(days)
    transition_errors: dict[str, float] = {}
    for cohort, matrix in instance.transitions.matrices.items():
        row_sums = matrix.df.groupby("from_state")["prob"].sum()
        transition_errors[cohort] = float((row_sums - 1.0).abs().max())

    hospital_ids = set(hospitals["hospital_id"].astype(str))
    allowed_lanes = lanes[lanes["allowed"].astype(int) == 1]
    max_possible_lanes = max(len(hospital_ids) * (len(hospital_ids) - 1), 1)
    transfer_density = float(len(allowed_lanes) / max_possible_lanes)

    checks = {
        "arrival_panel_complete": len(arrivals) == expected_arrival_rows,
        "elective_panel_complete": len(elective) == expected_elective_rows,
        "capacity_threshold_keys_aligned": set(
            map(tuple, capacities[["hospital_id", "resource"]].astype(str).to_numpy())
        )
        == set(map(tuple, safe[["hospital_id", "resource"]].astype(str).to_numpy())),
        "safe_thresholds_in_open_unit_interval": bool(
            ((safe["safe_utilization"] > 0) & (safe["safe_utilization"] <= 1)).all()
        ),
        "transition_rows_normalized": max(transition_errors.values(), default=0.0) <= 1e-9,
        "positive_arrival_volume": float(arrivals["arrivals"].sum()) > 0,
        "network_has_transfer_connectivity": len(allowed_lanes) > 0,
    }
    status = "passed" if all(checks.values()) else "failed"
    run_id = new_run_id("benchmark_readiness")
    report: dict[str, object] = {
        "run_id": run_id,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_evidence_basis": "synthetic_reference_case",
        "internal_calibration_status": "internally_validated_reference_benchmark",
        "external_historical_calibration_status": "not_supplied",
        "real_hospital_deployment_approval": False,
        "model_classification": "stochastic_state_transition_twin",
        "not_validated_as": "real_time_or_discrete_event_hospital_digital_twin",
        "reference_case_summary": {
            "hospital_count": int(len(hospitals)),
            "cohort_count": int(len(cohorts)),
            "elective_cohort_count": int(len(elective_cohorts)),
            "day_count": int(len(days)),
            "arrival_rows": int(len(arrivals)),
            "total_expected_arrivals": float(arrivals["arrivals"].sum()),
            "capacity_resource_rows": int(len(capacities)),
            "allowed_transfer_lanes": int(len(allowed_lanes)),
            "transfer_network_density": transfer_density,
            "safe_utilization_min": float(safe["safe_utilization"].min()),
            "safe_utilization_max": float(safe["safe_utilization"].max()),
            "maximum_transition_row_sum_error": float(
                max(transition_errors.values(), default=0.0)
            ),
        },
        "checks": checks,
        "future_calibration_event_contract": {
            "required_fields": [
                "event_timestamp",
                "hospital_id",
                "event_type",
                "patient_pathway_or_cohort",
                "resource",
                "quantity",
                "source_system",
            ],
            "minimum_event_types": [
                "arrival",
                "admission",
                "transfer",
                "discharge",
                "capacity_change",
                "staffing_change",
            ],
            "required_validation_targets": [
                "arrival_count_distribution",
                "occupancy_trajectory",
                "length_of_stay_or_transition_time",
                "transfer_flow",
                "waiting_time",
                "resource_utilization",
            ],
        },
        "claim_boundary": (
            "This gate validates internal coherence of the bundled synthetic benchmark. "
            "It does not establish external validity, clinical validity, or real-hospital calibration."
        ),
    }
    validation_dir = results_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    report_path = validation_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path, manifest_hash = write_run_manifest(
        run_id=run_id,
        run_type="benchmark_calibration_readiness",
        parameters={"evidence_basis": "synthetic_reference_case"},
        input_paths=[data_dir / "base_instance", data_dir / "transitions"],
        metrics={
            "passed_check_count": int(sum(checks.values())),
            "check_count": int(len(checks)),
            "total_expected_arrivals": float(arrivals["arrivals"].sum()),
            "maximum_transition_row_sum_error": float(
                max(transition_errors.values(), default=0.0)
            ),
        },
        notes=[report["claim_boundary"]],
        manifest_dir=Path(manifest_dir or (results_dir / "run_manifests")),
    )
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = manifest_hash
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_benchmark_calibration_readiness(), indent=2))
