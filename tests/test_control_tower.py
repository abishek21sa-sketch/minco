from __future__ import annotations

import json
from types import SimpleNamespace

from src.services.contract_registry import build_service_contract_registry
from src.services.control_tower_service import ControlTowerService


def test_control_tower_contract_is_registered_as_a_versioned_public_schema(tmp_path) -> None:
    output = tmp_path / "service_contract_registry.json"

    result = build_service_contract_registry(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result["status"] == "passed"
    assert "ControlTowerResponse" in result["schema_names"]
    assert "HealthResponse" in result["schema_names"]
    assert "ReadinessResponse" in result["schema_names"]
    assert "ReleaseReadinessResponse" in result["schema_names"]
    assert "ControlTowerResponse" in payload["schemas"]
    assert "ReleaseReadinessResponse" in payload["schemas"]
    assert payload["contract_version"] == "1.0"


def test_control_tower_is_human_gated_and_exposes_the_full_evidence_chain(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "base_instance").mkdir()
    (tmp_path / "transitions").mkdir()
    service = ControlTowerService(data_dir=tmp_path, results_dir=tmp_path)
    state = SimpleNamespace(
        state=SimpleNamespace(
            model_dump=lambda mode=None: {
                "snapshot_id": "state_test",
                "observed_at": "2026-09-04T12:00:00+00:00",
                "state_source": "synthetic_reference_case",
                "hospital_count": 2,
                "hospital_ids": ["H1", "H2"],
                "cohort_count": 1,
                "cohort_ids": ["c1"],
                "horizon_days": 1,
                "arrival_rows": 2,
                "total_expected_arrivals": 10.0,
                "capacity_by_resource": {"ICU": 20.0},
                "safe_utilization_min": 0.85,
                "safe_utilization_max": 0.90,
                "allowed_transfer_lanes": 2,
                "total_transfer_capacity_per_day": 8.0,
                "input_fingerprints": {"base_instance": "a" * 64, "transitions": "b" * 64},
            }
        )
    )
    monkeypatch.setattr(service.operational_state, "get_reference_snapshot", lambda context: state)
    monkeypatch.setattr(
        "src.services.control_tower_service.build_flow_cvar_decision",
        lambda: {
            "decision_id": "FLOW-TEST",
            "gate": "AUTHORIZED",
            "checks": {
                "optimizer_optimal": True,
                "solution_feasible": True,
                "markov_transition_row_stochastic": True,
            },
            "parameters": {"cvar_alpha": 0.8, "cvar_weight": 1.5},
            "decision": {
                "status": "OPTIMAL",
                "objective": 12.0,
                "expected_recourse_loss": 2.0,
                "tail_scenario_loss": 8.0,
                "cvar_loss": 8.0,
                "cvar_alpha": 0.8,
                "cvar_weight": 1.5,
                "bounded_claim": "not a clinical outcome guarantee",
            },
            "no_cvar_baseline": {"cvar_loss": 12.0},
            "flow_forecast": {
                "forecast_hospital_census": {"H1": 8.0, "H2": 5.0},
                "states": ["H1", "H2", "EXIT"],
                "current_state": [8.0, 5.0, 0.0],
                "transition_matrix": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.0, 0.0, 1.0]],
                "propagated_state": [7.0, 4.0, 1.0],
                "new_arrivals": [1.0, 1.0, 0.0],
                "next_state": [8.0, 5.0, 1.0],
                "row_sums": [1.0, 1.0, 1.0],
            },
            "actions": [{"action": "ACTIVATE_SURGE_BEDS", "hospital": "H1", "units": 2}],
            "scenario_actions": [],
            "scenario_probabilities": [0.8, 0.2],
            "operator_note": "Human review required.",
            "evidence_class": "synthetic algorithmic validation",
        },
    )
    monkeypatch.setattr(
        "src.services.control_tower_service.check_gurobi_readiness",
        lambda: SimpleNamespace(
            to_dict=lambda: {
                "package_available": False,
                "license_verified": False,
                "status": "package_unavailable",
                "version": None,
                "message": "not installed",
                "check_seconds": 0.0,
            }
        ),
    )

    payload = service.snapshot().model_dump(mode="json")

    assert payload["governance_state"] == "HUMAN_REVIEW"
    assert payload["governance"]["human_review_required"] is True
    assert payload["governance"]["autonomous_execution_permitted"] is False
    assert payload["decision"]["algorithm"] == "FLOW-CVaR"
    assert payload["risk"]["no_cvar_cvar_loss"] == 12.0
    assert payload["patient_flow"]["evidence_label"] == "PREDICTED"
    assert (
        payload["evidence"]["evidence_classes"]["realized"]
        == "Not present in the bundled evidence."
    )
    workflow = payload["governance"]["review_workflow"]
    assert workflow["immutable_event_log"] is True
    assert workflow["autonomous_execution_permitted"] is False
    assert "DEFERRED" in workflow["allowed_decisions"]
