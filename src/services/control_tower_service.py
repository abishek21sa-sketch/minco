"""Enterprise control-tower aggregation for MINCO.

This service deliberately composes existing MINCO services and the governed
FLOW-CVaR signature algorithm.  It is an operator-facing read model, not a
new decision engine and not an authorization to execute hospital actions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.paths import AUDIT_DB_PATH, DATA_DIR, RESULTS_DIR
from src.contracts.common import (
    REFERENCE_CASE,
    REFERENCE_CASE_BOUNDARY,
    ExecutionContext,
    new_correlation_id,
    utc_now,
)
from src.contracts.control_tower import ControlTowerResponse
from src.contracts.operational_history import OperationalHistoryResponse
from src.contracts.review import REVIEW_DECISIONS
from src.decision_math.flow_cvar_decision import build_flow_cvar_decision
from src.services.operational_state_service import OperationalStateService
from src.services.operational_history_service import OperationalHistoryService
from src.services.scenario_service import BUILT_IN_SCENARIOS
from src.storage.audit_repository import read_recent_decision_reviews, verify_decision_review_chain
from src.validation.run_manifest import describe_path, sha256_file
from src.validation.solver_readiness import check_gurobi_readiness


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _severity(level: str) -> int:
    return {"GREEN": 0, "YELLOW": 1, "RED": 2}.get(level, 1)


def _highest_level(*levels: str) -> str:
    return max((level for level in levels if level in {"GREEN", "YELLOW", "RED"}), key=_severity, default="YELLOW")


def _freshness(status: dict[str, Any], now: datetime) -> dict[str, Any]:
    timestamp = _parse_timestamp(status.get("timestamp"))
    if timestamp is None:
        return {
            "status": "UNAVAILABLE",
            "observed_at": None,
            "age_hours": None,
            "service_level_hours": 24,
            "note": "No timestamped operational status artifact is available.",
        }
    age_hours = max(0.0, (now - timestamp).total_seconds() / 3600.0)
    freshness_status = "CURRENT" if age_hours <= 24 else "STALE"
    return {
        "status": freshness_status,
        "observed_at": timestamp.isoformat(),
        "age_hours": round(age_hours, 2),
        "service_level_hours": 24,
        "note": (
            "Timestamped status is within the 24-hour operating freshness target."
            if freshness_status == "CURRENT"
            else "Status is older than the 24-hour operating freshness target and requires operator confirmation."
        ),
    }


def _artifact(path: Path, label: str, evidence_class: str) -> dict[str, Any]:
    description = describe_path(path)
    return {
        "label": label,
        "evidence_class": evidence_class,
        **description,
    }


def _status_alert(status: dict[str, Any], freshness: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    state = status.get("state") if isinstance(status.get("state"), dict) else {}
    reported = str(status.get("overall_alert_level", "")).upper()
    if reported not in {"RED", "YELLOW", "GREEN"}:
        max_utilization = float(state.get("max_utilization_ratio", 0.0) or 0.0)
        unsafe_excess = float(state.get("total_unsafe_excess", 0.0) or 0.0)
        reported = "RED" if max_utilization >= 1.10 or unsafe_excess >= 10 else (
            "YELLOW" if max_utilization >= 1.0 or unsafe_excess > 0 else "GREEN"
        )

    alerts: list[dict[str, Any]] = []
    if reported != "GREEN":
        alerts.append(
            {
                "level": reported,
                "code": "NETWORK_PRESSURE",
                "message": "The reference status artifact reports operational pressure that requires review.",
                "evidence_class": "SIMULATED",
            }
        )
    if freshness["status"] in {"STALE", "UNAVAILABLE"}:
        alerts.append(
            {
                "level": "YELLOW",
                "code": "STATUS_FRESHNESS",
                "message": freshness["note"],
                "evidence_class": "OBSERVED_ARTIFACT_METADATA",
            }
        )
    return reported, alerts


class ControlTowerService:
    """Compose an enterprise-grade, read-only decision snapshot."""

    service_name = "control_tower_service"

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        results_dir: Path = RESULTS_DIR,
        operational_state: OperationalStateService | None = None,
        db_path: Path = AUDIT_DB_PATH,
        operational_history: OperationalHistoryService | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.operational_state = operational_state or OperationalStateService(self.data_dir)
        self.db_path = Path(db_path)
        self.operational_history = operational_history

    def snapshot(self, context: ExecutionContext | None = None) -> ControlTowerResponse:
        context = context or ExecutionContext(correlation_id=new_correlation_id(), source="service")
        generated_at = utc_now()
        state_response = self.operational_state.get_reference_snapshot(context)
        state = state_response.state.model_dump(mode="json")

        if self.operational_history is not None:
            history_payload = self.operational_history.snapshot(context).model_dump(mode="json")
        else:
            history_payload = OperationalHistoryResponse(
                generated_at=generated_at,
                correlation_id=context.correlation_id,
                data_mode="UNAVAILABLE",
                data_stack="NOT_CONFIGURED",
                event_count=0,
                as_of=generated_at,
                freshness_status="UNAVAILABLE",
                freshness={"target_hours": 24, "note": "Operational history service not configured."},
                event_type_counts={},
                source_modes=[],
                hospital_ids=[],
                reconstructed_state={},
                evidence={"evidence_label": "UNAVAILABLE"},
                warnings=["Operational history service is not configured for this control-tower instance."],
            ).model_dump(mode="json")

        flow_payload = build_flow_cvar_decision()
        decision = flow_payload["decision"]
        no_cvar = flow_payload["no_cvar_baseline"]
        status_path = self.results_dir / "command_center" / "current_status.json"
        status = _read_json(status_path)
        freshness = _freshness(status, generated_at)
        reported_alert, alerts = _status_alert(status, freshness)

        checks = {str(key): bool(value) for key, value in flow_payload.get("checks", {}).items()}
        inputs_present = all(
            (self.data_dir / name).exists() for name in ("base_instance", "transitions")
        )
        checks["input_tables_present"] = inputs_present
        checks["synthetic_case_explicitly_labeled"] = True
        checks["human_review_required"] = True
        checks["autonomous_execution_disabled"] = True
        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        governance_state = "HUMAN_REVIEW" if not failed_checks else "BLOCKED"
        if failed_checks:
            alerts.append(
                {
                    "level": "RED",
                    "code": "EVIDENCE_GATE",
                    "message": f"Control-tower evidence gate failed: {', '.join(failed_checks)}.",
                    "evidence_class": "OPTIMIZED",
                }
            )

        solver = check_gurobi_readiness().to_dict()
        artifact_paths = [
            _artifact(self.data_dir / "base_instance", "base_instance", "SYNTHETIC_REFERENCE_CASE"),
            _artifact(self.data_dir / "transitions", "patient_flow_transitions", "SYNTHETIC_REFERENCE_CASE"),
        ]
        product_evidence_path = (
            self.results_dir.parent / "artifacts" / "product_runtime" / "latest_product_evidence.json"
        )
        if product_evidence_path.exists():
            artifact_paths.append(_artifact(product_evidence_path, "latest_product_evidence", "OPTIMIZED"))
        registry_path = self.results_dir / "validation" / "data_asset_registry_summary.json"
        if registry_path.exists():
            artifact_paths.append(_artifact(registry_path, "data_asset_registry_summary", "GOVERNANCE"))

        first_stage_actions = list(flow_payload.get("actions", []))
        scenario_actions = list(flow_payload.get("scenario_actions", []))
        risk = {
            "expected_recourse_loss": float(decision.get("expected_recourse_loss", 0.0)),
            "tail_scenario_loss": float(decision.get("tail_scenario_loss", 0.0)),
            "cvar_loss": float(decision.get("cvar_loss", 0.0)),
            "cvar_alpha": float(decision.get("cvar_alpha", flow_payload["parameters"]["cvar_alpha"])),
            "cvar_weight": float(decision.get("cvar_weight", flow_payload["parameters"]["cvar_weight"])),
            "no_cvar_cvar_loss": float(no_cvar.get("cvar_loss", 0.0)),
            "delta_vs_no_cvar": round(float(decision.get("cvar_loss", 0.0)) - float(no_cvar.get("cvar_loss", 0.0)), 6),
            "evidence_label": "OPTIMIZED",
            "scenario_count": len(flow_payload.get("scenario_probabilities", [])),
            "scenario_probabilities": flow_payload.get("scenario_probabilities", []),
        }

        network = {
            **state,
            "projected_census": flow_payload["flow_forecast"]["forecast_hospital_census"],
            "modeled_hospitals": sorted(flow_payload["flow_forecast"]["forecast_hospital_census"]),
            "state_inventory_hospitals": state["hospital_ids"],
            "model_scope_note": (
                "The operational inventory contains the bundled H1-H3 reference network; the governed "
                "FLOW-CVaR signature benchmark currently solves the H1-H2 two-hospital formulation."
            ),
            "evidence_label": "OBSERVED_ARTIFACT_METADATA",
        }

        audit_integrity = verify_decision_review_chain(self.db_path)

        governance = {
            "state": governance_state,
            "model_gate": flow_payload.get("gate", "BLOCKED"),
            "release_readiness": "NOT_FOR_PRODUCTION",
            "human_review_required": True,
            "autonomous_execution_permitted": False,
            "authorization_scope": "Model and evidence readiness only; hospital operations approval is required.",
            "blocking_reasons": [
                "Bundled data is synthetic reference-case evidence, not a live hospital feed.",
                "The current status artifact is not an authorization to execute staffing, transfer, diversion, or clinical actions.",
                *(["Operational status freshness requires confirmation."] if freshness["status"] != "CURRENT" else []),
            ],
            "failed_checks": failed_checks,
            "gate_checks": checks,
            "operator_action": "Reconcile the recommendation with current hospital command-center facts before any operational decision.",
            "claim_boundary": REFERENCE_CASE_BOUNDARY,
            "review_workflow": {
                "review_endpoint": "/v1/audit/{run_id}/review",
                "history_endpoint": "/v1/audit/{run_id}/reviews",
                "recent_reviews_endpoint": "/v1/audit/reviews",
                "integrity_endpoint": "/v1/audit/integrity",
                "integrity_status": audit_integrity["status"],
                "integrity_review_count": audit_integrity["review_count"],
                "allowed_decisions": list(REVIEW_DECISIONS),
                "immutable_event_log": True,
                "autonomous_execution_permitted": False,
                "effect": (
                    "Records a human disposition of a persisted recommendation only; it never dispatches "
                    "staffing, transfer, diversion, or clinical actions."
                ),
            },
        }

        diagnostics = {
            "reference_solver_backend": "SciPy/HiGHS independent small-instance MILP oracle",
            "production_solver_preflight": solver,
            "solver_status": decision.get("status"),
            "objective_value": decision.get("objective"),
            "is_mip": True,
            "cvar_alpha": risk["cvar_alpha"],
            "cvar_weight": risk["cvar_weight"],
            "scenario_count": risk["scenario_count"],
            "hospitals_modeled": len(network["modeled_hospitals"]),
            "periods_modeled": 1,
            "decision_variable_groups": [
                "surge_beds",
                "flex_staff_blocks",
                "diversion_or_deferral",
                "transfers",
                "scenario_overflow",
                "cvar_eta_and_excess",
            ],
            "feasibility_checks": checks,
        }

        evidence = {
            "reference_case": REFERENCE_CASE,
            "evidence_classes": {
                "observed": "Static synthetic reference-case tables and timestamped generated status artifact.",
                "predicted": "Markov patient-flow forecast.",
                "simulated": "Reference and rare-surge scenario demand.",
                "optimized": "FLOW-CVaR integer capacity recommendation.",
                "realized": "Not present in the bundled evidence.",
            },
            "artifacts": artifact_paths,
            "input_fingerprints": state["input_fingerprints"],
            "latest_product_evidence_sha256": (
                sha256_file(product_evidence_path) if product_evidence_path.exists() else None
            ),
            "data_freshness": freshness,
            "claim_boundary": REFERENCE_CASE_BOUNDARY,
        }

        scenarios = [
            {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "description": scenario.description,
                "parameters": scenario.parameters.model_dump(mode="json"),
                "built_in": scenario.built_in,
            }
            for scenario in BUILT_IN_SCENARIOS
        ]
        recent_reviews = read_recent_decision_reviews(n=5, db_path=self.db_path)
        governance["review_workflow"]["recent_review_count"] = len(recent_reviews)
        governance["review_workflow"]["latest_review"] = (
            recent_reviews[0] if recent_reviews else None
        )

        overall_level = _highest_level(reported_alert, *(str(item["level"]) for item in alerts))
        decision_view = {
            "decision_id": flow_payload.get("decision_id"),
            "algorithm": "FLOW-CVaR",
            "gate": flow_payload.get("gate"),
            "human_review_required": True,
            "execution_mode": "RECOMMENDATION_ONLY",
            "first_stage_actions": first_stage_actions,
            "scenario_recourse_actions": scenario_actions,
            "selected_action_count": len(first_stage_actions) + len(scenario_actions),
            "bounded_claim": decision.get("bounded_claim"),
            "operator_note": flow_payload.get("operator_note"),
        }

        return ControlTowerResponse(
            snapshot_id=f"tower_{context.correlation_id.removeprefix('corr_')[:24]}",
            generated_at=generated_at,
            correlation_id=context.correlation_id,
            overall_alert_level=overall_level,
            governance_state=governance_state,
            governance=governance,
            network=network,
            patient_flow={
                **flow_payload["flow_forecast"],
                "evidence_label": "PREDICTED",
                "observed_vs_predicted_note": "The Markov forecast is model-predicted flow, not causal proof or realized census.",
            },
            risk=risk,
            decision=decision_view,
            diagnostics=diagnostics,
            evidence=evidence,
            operational_history=history_payload,
            scenario_catalog=scenarios,
            alerts=alerts,
        )
