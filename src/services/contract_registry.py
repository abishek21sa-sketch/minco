"""Generate a hash-addressed registry of public service schemas."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.paths import RESULTS_DIR
from src.contracts.common import CONTRACT_VERSION
from src.contracts.control_tower import ControlTowerResponse
from src.contracts.decision_packet import DecisionPacketResponse
from src.contracts.identity import IdentityResponse
from src.contracts.decision import (
    DecisionRecommendationRequest,
    DecisionRecommendationResponse,
)
from src.contracts.operational_state import OperationalStateResponse
from src.contracts.operational_history import OperationalHistoryResponse
from src.contracts.event_ingestion import (
    EventIngestionRequest,
    EventIngestionResponse,
    EventIngestionStatusResponse,
)
from src.contracts.release_readiness import ReleaseReadinessResponse
from src.contracts.release_evidence import ReleaseEvidenceResponse
from src.contracts.review import AuditIntegrityResponse, DecisionReviewRequest, DecisionReviewResponse
from src.contracts.scenario import ScenarioEvaluationRequest, ScenarioEvaluationResponse
from src.contracts.system import HealthResponse, ReadinessResponse
from src.contracts.observability import ObservabilityMetricsResponse
from src.validation.run_manifest import sha256_file


CONTRACT_REGISTRY_DIR = RESULTS_DIR / "contracts"
CONTRACT_REGISTRY_PATH = CONTRACT_REGISTRY_DIR / "service_contract_registry.json"


def build_service_contract_registry(
    output_path: Path = CONTRACT_REGISTRY_PATH,
) -> dict[str, object]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = {
        "OperationalStateResponse": OperationalStateResponse,
        "OperationalHistoryResponse": OperationalHistoryResponse,
        "EventIngestionRequest": EventIngestionRequest,
        "EventIngestionResponse": EventIngestionResponse,
        "EventIngestionStatusResponse": EventIngestionStatusResponse,
        "ReleaseReadinessResponse": ReleaseReadinessResponse,
        "ReleaseEvidenceResponse": ReleaseEvidenceResponse,
        "ScenarioEvaluationRequest": ScenarioEvaluationRequest,
        "ScenarioEvaluationResponse": ScenarioEvaluationResponse,
        "DecisionRecommendationRequest": DecisionRecommendationRequest,
        "DecisionRecommendationResponse": DecisionRecommendationResponse,
        "ControlTowerResponse": ControlTowerResponse,
        "DecisionPacketResponse": DecisionPacketResponse,
        "IdentityResponse": IdentityResponse,
        "DecisionReviewRequest": DecisionReviewRequest,
        "DecisionReviewResponse": DecisionReviewResponse,
        "AuditIntegrityResponse": AuditIntegrityResponse,
        "HealthResponse": HealthResponse,
        "ReadinessResponse": ReadinessResponse,
        "ObservabilityMetricsResponse": ObservabilityMetricsResponse,
    }
    payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schemas": {name: model.model_json_schema() for name, model in models.items()},
        "compatibility_policy": (
            "Versioned /v1 contracts are additive within contract version 1.0. Breaking field or "
            "semantic changes require a new API contract version."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": "passed",
        "contract_version": CONTRACT_VERSION,
        "schema_count": len(models),
        "schema_names": sorted(models),
        "registry_path": str(output_path),
        "registry_sha256": sha256_file(output_path),
    }
