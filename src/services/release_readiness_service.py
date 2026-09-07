"""Release-readiness scorecard composed from current local evidence artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.paths import RESULTS_DIR
from src.contracts.common import ExecutionContext, REFERENCE_CASE_BOUNDARY, new_correlation_id, utc_now
from src.contracts.release_readiness import ReleaseReadinessResponse
from src.observability.security import ROLE_LEVELS, api_key_enforcement_enabled, configured_credentials
from src.services.release_evidence_service import ReleaseEvidenceService
from src.validation.run_manifest import describe_path
from src.validation.solver_readiness import check_gurobi_readiness


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


class ReleaseReadinessService:
    """Report whether the build is internally evidenced and what blocks production."""

    service_name = "release_readiness_service"

    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        self.results_dir = Path(results_dir)

    def snapshot(self, context: ExecutionContext | None = None) -> ReleaseReadinessResponse:
        context = context or ExecutionContext(
            correlation_id=new_correlation_id(), source="service"
        )
        generated_at = utc_now()
        registry_path = self.results_dir / "contracts" / "service_contract_registry.json"
        product_evidence_path = (
            self.results_dir.parent / "artifacts" / "product_runtime" / "latest_product_evidence.json"
        )
        phase2_path = self.results_dir / "validation" / "phase2_backend_acceptance_report.json"
        phase2 = _read_json(phase2_path)
        solver = check_gurobi_readiness().to_dict()
        authentication_enforced = api_key_enforcement_enabled()
        configured_identity_count = len(configured_credentials())
        release_evidence = ReleaseEvidenceService(self.results_dir).snapshot(context)

        checks = {
            "contract_registry_present": registry_path.exists(),
            "product_runtime_acceptance_evidence_present": product_evidence_path.exists(),
            "phase2_backend_gate_passed": phase2.get("status") == "passed",
            "synthetic_case_explicitly_labeled": True,
            "human_review_required": True,
            "autonomous_execution_disabled": True,
            "production_identity_access_control_configured": authentication_enforced,
            "role_based_authorization_configured": configured_identity_count > 0,
            "live_adt_ehr_feed_connected": False,
            "production_solver_license_verified": bool(solver["license_verified"]),
            "release_evidence_attestation_verified": release_evidence.status == "VERIFIED",
        }
        blocking_reasons = [
            "The current evidence base is the synthetic Meridian reference case, not a validated production cohort.",
            "No live ADT/EHR operational feed is connected to this build.",
            "Production identity and access control are not configured in the local reference package."
            if not authentication_enforced
            else "",
            "The production Gurobi license is not verified in this runtime."
            if not solver["license_verified"]
            else "",
        ]
        blocking_reasons = [reason for reason in blocking_reasons if reason]
        next_actions = [
            "Connect an approved ADT/EHR or event-stream source and complete data-contract validation.",
            "Configure enterprise identity/access control and rotate secrets through the deployment platform.",
            "Complete external hospital validation and compare predicted versus realized outcomes.",
        ]
        if not solver["license_verified"]:
            next_actions.append("Install and verify the approved production solver environment, or document the governed fallback.")

        return ReleaseReadinessResponse(
            generated_at=generated_at,
            correlation_id=context.correlation_id,
            status="CONDITIONAL" if all(checks.values()) else "BLOCKED",
            checks=checks,
            dependency_status={
                "solver": solver,
                "phase2_backend": {
                    "status": phase2.get("status", "UNAVAILABLE"),
                    "event_count": phase2.get("event_count"),
                    "plan_solver": phase2.get("plan_solver"),
                    "julia_readiness": phase2.get("julia_readiness"),
                },
            },
            security_controls={
                "api_key_configured": authentication_enforced,
                "authentication_enforced": authentication_enforced,
                "security_mode": "API_KEY_ENFORCED" if authentication_enforced else "LOCAL_REFERENCE_CASE_NO_AUTH",
                "configured_identity_count": configured_identity_count,
                "role_hierarchy": ROLE_LEVELS,
                "review_minimum_role": "reviewer",
                "correlation_header_enabled": True,
                "sensitive_payload_logging_disabled": True,
            },
            evidence_artifacts={
                "contract_registry": describe_path(registry_path),
                "product_runtime_evidence": describe_path(product_evidence_path),
                "phase2_backend_acceptance": describe_path(phase2_path),
                "release_evidence_attestation": {
                    "status": release_evidence.status,
                    "build_fingerprint": release_evidence.build_fingerprint,
                },
            },
            blocking_reasons=blocking_reasons,
            next_actions=next_actions,
            claim_boundary=REFERENCE_CASE_BOUNDARY,
        )
