"""Level 3 WP1 integration and contract validation gate."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config.paths import RESULTS_DIR
from src.contracts.common import ExecutionContext
from src.contracts.decision import DecisionRecommendationRequest
from src.contracts.scenario import ScenarioEvaluationRequest
from src.services.container import build_service_container
from src.services.contract_registry import build_service_contract_registry
from src.storage.audit_repository import get_recent_audit_runs
from src.validation.run_manifest import new_run_id, write_run_manifest
from src.version import __version__


LEVEL2_REPORT = RESULTS_DIR / "validation" / "level2_completion_report.json"
REPORT_PATH = RESULTS_DIR / "validation" / "level3_wp1_validation_report.json"


def run_level3_wp1_validation() -> dict[str, object]:
    if not LEVEL2_REPORT.exists():
        raise FileNotFoundError(
            "Level 2 completion evidence is missing. Run the Level 2 completion gate first."
        )
    level2 = json.loads(LEVEL2_REPORT.read_text(encoding="utf-8"))
    if level2.get("status") != "passed" or level2.get("level_2_disposition") != (
        "complete_and_level_3_authorized"
    ):
        raise RuntimeError("Level 2 has not authorized Level 3 execution.")

    services = build_service_container()
    context = ExecutionContext(source="level3_wp1_validation")
    contract_registry = build_service_contract_registry()
    state_response = services.operational_state.get_reference_snapshot(context)
    scenario_catalog = services.scenarios.list_scenarios()

    scenario_response = services.scenarios.evaluate(
        ScenarioEvaluationRequest(
            scenario_id="baseline",
            n_replications=3,
            context=context,
        )
    )
    decision_response = services.decisions.recommend(
        DecisionRecommendationRequest(
            scenario_id="baseline",
            scenario_label="Level 3 WP1 integration baseline",
            n_replications=3,
            context=context,
        )
    )

    recent = get_recent_audit_runs(n=100)
    audited_run_ids = {str(row["run_id"]) for row in recent}
    audit_checks = {
        "scenario_run_present": scenario_response.run_id in audited_run_ids,
        "decision_run_present": decision_response.run_id in audited_run_ids,
        "scenario_manifest_attached": scenario_response.metadata.evidence is not None,
        "decision_manifest_attached": decision_response.metadata.evidence is not None,
    }
    checks = {
        "level2_authorization_present": True,
        "contract_registry_generated": contract_registry["status"] == "passed",
        "operational_state_contract_valid": state_response.state.hospital_count == 3,
        "scenario_catalog_available": len(scenario_catalog) >= 5,
        "scenario_service_executed": scenario_response.metrics.model_status == 2,
        "decision_service_executed": decision_response.metrics.model_status == 2,
        "human_review_required": decision_response.recommendation.human_review_required,
        **audit_checks,
    }
    status = "passed" if all(checks.values()) else "failed"

    run_id = new_run_id("level3_wp1")
    report: dict[str, object] = {
        "work_package": "level_3_wp1_operational_state_scenario_and_service_contracts",
        "version": __version__,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "level_3_wp1_disposition": (
            "complete_and_wp2_authorized" if status == "passed" else "not_complete"
        ),
        "checks": checks,
        "contract_registry": contract_registry,
        "operational_state_summary": {
            "snapshot_id": state_response.state.snapshot_id,
            "hospital_count": state_response.state.hospital_count,
            "cohort_count": state_response.state.cohort_count,
            "horizon_days": state_response.state.horizon_days,
            "total_expected_arrivals": state_response.state.total_expected_arrivals,
            "state_source": state_response.state.state_source,
        },
        "scenario_catalog": [item.scenario_id for item in scenario_catalog],
        "integration_runs": {
            "scenario_run_id": scenario_response.run_id,
            "decision_run_id": decision_response.run_id,
            "overall_alert_level": decision_response.overall_alert_level,
            "recommended_action": decision_response.recommendation.action_id,
        },
        "claim_boundary": state_response.metadata.claim_boundary,
        "required_follow_up": [] if status == "passed" else [
            key for key, value in checks.items() if not value
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest_path, manifest_hash = write_run_manifest(
        run_id=run_id,
        run_type="level3_wp1_integration_gate",
        parameters={"n_replications": 3, "contract_version": "1.0"},
        input_paths=[LEVEL2_REPORT],
        output_paths=[RESULTS_DIR / "contracts" / "service_contract_registry.json"],
        metrics={"status": status, **checks},
        model_info={
            "architecture": "typed service contracts and application-service orchestration",
            "simulation": "stochastic state-transition twin",
            "optimizer": "continuous network capacity LP",
        },
        notes=[state_response.metadata.claim_boundary],
    )
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = manifest_hash
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(run_level3_wp1_validation(), indent=2))


if __name__ == "__main__":
    main()
