"""FastAPI application for the MINCO decision-intelligence platform.

Versioned endpoints delegate to application services and typed contracts.
Legacy endpoints remain backward compatible while using the same service layer.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse, PlainTextResponse

from src.api.ui import control_tower_html
from src.config.paths import RESULTS_DIR
from src.contracts.common import ExecutionContext
from src.contracts.control_tower import ControlTowerResponse
from src.contracts.decision import (
    DecisionRecommendationRequest,
    DecisionRecommendationResponse,
)
from src.contracts.decision_packet import DecisionPacketResponse
from src.contracts.event_ingestion import (
    EventIngestionRequest,
    EventIngestionResponse,
    EventIngestionStatusResponse,
)
from src.contracts.identity import IdentityResponse
from src.contracts.observability import ObservabilityMetricsResponse
from src.contracts.operational_history import OperationalHistoryResponse
from src.contracts.operational_state import OperationalStateResponse
from src.contracts.release_evidence import ReleaseEvidenceResponse
from src.contracts.release_readiness import ReleaseReadinessResponse
from src.contracts.review import (
    AuditIntegrityResponse,
    DecisionReviewRequest,
    DecisionReviewResponse,
)
from src.contracts.scenario import (
    ScenarioDefinition,
    ScenarioEvaluationRequest,
    ScenarioEvaluationResponse,
    ScenarioParameters,
)
from src.contracts.surge_plan import SurgePlanRequest, SurgePlanResponse
from src.contracts.system import HealthResponse, ReadinessResponse
from src.decision_math.flow_cvar_decision import build_flow_cvar_decision
from src.observability.metrics import API_METRICS
from src.observability.security import (
    ROLE_LEVELS,
    api_key_enforcement_enabled,
    api_key_middleware,
    identity_for_request,
)
from src.observability.telemetry import correlation_id_for_request, correlation_middleware
from src.services.container import ServiceContainer, build_service_container
from src.services.decision_packet_service import DecisionPacketNotFoundError
from src.services.scenario_service import ScenarioNotFoundError
from src.services.surge_planning_service import SurgePlanningNotReadyError
from src.storage.audit_repository import (
    AuditRunNotFoundError,
    get_decision_run,
    get_recent_alerts,
    get_recent_audit_runs,
    get_recent_decision_reviews,
    get_reviews_for_run,
    log_decision_review,
    verify_decision_review_chain,
)
from src.validation.solver_readiness import check_gurobi_readiness
from src.version import __version__

COMMAND_CENTER_STATUS_PATH = RESULTS_DIR / "command_center" / "current_status.json"
PREDICTIVE_GOVERNANCE_PATH = RESULTS_DIR / "validation" / "demand_forecast_validation.json"
LEVEL2_COMPLETION_PATH = RESULTS_DIR / "validation" / "level2_completion_report.json"
DATA_ASSET_REGISTRY_SUMMARY_PATH = RESULTS_DIR / "validation" / "data_asset_registry_summary.json"
LEVEL3_WP1_REPORT_PATH = RESULTS_DIR / "validation" / "level3_wp1_validation_report.json"
FINALIZATION_PHASE1_REPORT_PATH = (
    RESULTS_DIR / "validation" / "finalization_phase1_validation_report.json"
)


class WhatIfRequest(BaseModel):
    """Backward-compatible request for POST /what-if."""

    icu_bed_delta: int = Field(default=0, description="ICU beds to add or remove network-wide")
    transfer_capacity_multiplier: float = Field(default=1.0, ge=0.0, le=5.0)
    demand_surge_multiplier: float = Field(default=1.0, ge=0.0, le=5.0)
    n_replications: int = Field(default=10, ge=1, le=200)
    surface: str = Field(default="api")


class EventRequest(BaseModel):
    event_name: str
    n_replications: int = Field(default=10, ge=1, le=200)


class RecommendRequest(BaseModel):
    scenario_label: str = Field(default="current network state")
    n_replications: int = Field(default=50, ge=1, le=200)


def _solver_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "code": "solver_runtime_unavailable",
            "message": "This endpoint requires the optional Gurobi runtime and a valid license.",
            "cause": f"{type(exc).__name__}: {exc}",
        },
    )


def _translate_service_exception(exc: Exception) -> None:
    if isinstance(exc, ScenarioNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, SurgePlanningNotReadyError):
        raise HTTPException(
            status_code=503,
            detail={"code": "surge_planning_not_ready", "message": str(exc)},
        ) from exc
    module_name = type(exc).__module__
    if isinstance(exc, (ImportError, ModuleNotFoundError, RuntimeError)) or module_name.startswith(
        "gurobipy"
    ):
        raise _solver_unavailable(exc) from exc
    raise exc


def _legacy_whatif_payload(response: ScenarioEvaluationResponse) -> dict[str, Any]:
    evidence = response.metadata.evidence
    return {
        "run_id": response.run_id,
        "manifest": None if evidence is None else evidence.model_dump(),
        **response.metrics.model_dump(),
    }


def _legacy_recommendation_payload(
    response: DecisionRecommendationResponse,
) -> dict[str, Any]:
    evidence = response.metadata.evidence
    return {
        "run_id": response.run_id,
        "overall_alert_level": response.overall_alert_level,
        "state": response.metrics.model_dump(),
        "alerts": [item.model_dump() for item in response.alerts],
        "recommendation": response.recommendation.model_dump(),
        "executive_brief": response.executive_brief,
        "manifest": None if evidence is None else evidence.model_dump(),
    }


def _request_context(http_request: Request, source: str) -> ExecutionContext:
    return ExecutionContext(
        correlation_id=correlation_id_for_request(http_request),
        source=source,
    )


def _propagate_correlation(payload: Any, http_request: Request) -> Any:
    """Keep caller trace identity aligned with typed service response metadata."""
    return payload.model_copy(
        update={
            "context": payload.context.model_copy(
                update={"correlation_id": correlation_id_for_request(http_request)}
            )
        }
    )


def create_app(services: ServiceContainer | None = None) -> FastAPI:
    services = services or build_service_container()
    app = FastAPI(
        title="MINCO — Healthcare Operations Decision Intelligence API",
        description=(
            "Service boundary for the MINCO platform. The bundled Meridian Health Network "
            "is a synthetic reference case, not a real health system."
        ),
        version=__version__,
    )

    origins = [
        value.strip()
        for value in os.getenv("MINCO_CORS_ORIGINS", "http://localhost:8501").split(",")
        if value.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(api_key_middleware)
    app.middleware("http")(correlation_middleware)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def control_tower_ui() -> HTMLResponse:
        """Serve the executive Command Center shell backed by versioned APIs."""
        return HTMLResponse(control_tower_html())

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health(request: Request) -> HealthResponse:
        return HealthResponse(
            service="minco-api",
            version=__version__,
            correlation_id=correlation_id_for_request(request),
        )

    @app.get("/readiness", response_model=ReadinessResponse, tags=["System"])
    def readiness(request: Request) -> ReadinessResponse:
        data_dir = services.operational_state.data_dir
        data_ready = (data_dir / "base_instance").exists() and (data_dir / "transitions").exists()
        solver = check_gurobi_readiness().to_dict()
        fully_ready = data_ready and solver["license_verified"]
        checks = {
            "data_assets_present": data_ready,
            "solver_package_available": bool(solver["package_available"]),
            "solver_license_verified": bool(solver["license_verified"]),
            "synthetic_reference_case_only": True,
            "autonomous_execution_disabled": True,
            "api_key_enforced_for_operational_routes": api_key_enforcement_enabled(),
        }
        return ReadinessResponse(
            status="ready" if fully_ready else "degraded" if data_ready else "not_ready",
            service="minco-api",
            version=__version__,
            correlation_id=correlation_id_for_request(request),
            data_ready=data_ready,
            solver_package_available=bool(solver["package_available"]),
            solver_license_verified=bool(solver["license_verified"]),
            solver_status=str(solver["status"]),
            solver_version=solver["version"],
            solver_check_seconds=float(solver["check_seconds"]),
            checks=checks,
            note=str(solver["message"]),
        )

    @app.get(
        "/v1/identity",
        response_model=IdentityResponse,
        tags=["V1 Governance"],
    )
    def identity(request: Request) -> IdentityResponse:
        """Return the redacted caller identity and effective role posture."""
        caller = identity_for_request(request)
        return IdentityResponse(
            subject=caller.subject,
            role=caller.role,
            auth_method=caller.auth_method,
            authenticated=caller.authenticated,
            authorization_enforced=api_key_enforcement_enabled(),
            permissions=list(caller.permissions),
            role_hierarchy=ROLE_LEVELS,
            correlation_id=correlation_id_for_request(request),
        )

    @app.get(
        "/v1/observability/metrics",
        response_model=ObservabilityMetricsResponse,
        tags=["V1 Observability"],
    )
    def observability_metrics(request: Request) -> ObservabilityMetricsResponse:
        """Return low-cardinality request metrics without payload or ID retention."""
        return ObservabilityMetricsResponse(
            generated_at=datetime.now(UTC),
            correlation_id=correlation_id_for_request(request),
            **API_METRICS.snapshot(),
            data_logging_policy={
                "query_strings_logged": False,
                "request_payloads_logged": False,
                "patient_identifiers_retained": False,
                "route_labels_normalized": True,
            },
        )

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    def prometheus_metrics() -> PlainTextResponse:
        """Expose Prometheus-compatible low-cardinality metrics for platform scrapers."""
        return PlainTextResponse(API_METRICS.prometheus(), media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------------
    # Versioned Level 3 service endpoints
    # ------------------------------------------------------------------
    @app.get(
        "/v1/operational-state",
        response_model=OperationalStateResponse,
        tags=["V1 Operational State"],
    )
    def operational_state(request: Request) -> OperationalStateResponse:
        return services.operational_state.get_reference_snapshot(
            _request_context(request, "api_v1_operational_state")
        )

    @app.get(
        "/v1/control-tower",
        response_model=ControlTowerResponse,
        tags=["V1 Control Tower"],
    )
    def control_tower(request: Request) -> ControlTowerResponse:
        """Return one auditable operator snapshot across MINCO's decision chain."""
        return services.control_tower.snapshot(_request_context(request, "api_v1_control_tower"))

    @app.get(
        "/v1/control-tower/snapshot",
        response_model=ControlTowerResponse,
        include_in_schema=False,
    )
    def control_tower_snapshot(request: Request) -> ControlTowerResponse:
        """Compatibility alias for clients that name the resource explicitly."""
        return services.control_tower.snapshot(
            _request_context(request, "api_v1_control_tower_snapshot")
        )

    @app.get(
        "/v1/operational-history",
        response_model=OperationalHistoryResponse,
        tags=["V1 Operational History"],
    )
    def operational_history(request: Request) -> OperationalHistoryResponse:
        """Return replay/event-lake provenance and reconstructed state freshness."""
        return services.operational_history.snapshot(
            _request_context(request, "api_v1_operational_history")
        )

    @app.get(
        "/v1/operational-events/status",
        response_model=EventIngestionStatusResponse,
        tags=["V1 Operational Events"],
    )
    def operational_events_status(request: Request) -> EventIngestionStatusResponse:
        """Return external-event gateway posture without claiming feed connectivity."""
        return services.live_ingestion.status(
            _request_context(request, "api_v1_operational_events_status")
        )

    @app.post(
        "/v1/operational-events/ingest",
        response_model=EventIngestionResponse,
        tags=["V1 Operational Events"],
    )
    def ingest_operational_events(
        payload: EventIngestionRequest,
        request: Request,
    ) -> EventIngestionResponse:
        """Validate and append a bounded external event batch; never execute actions."""
        return services.live_ingestion.ingest(
            payload,
            _request_context(request, "api_v1_operational_events_ingest"),
        )

    @app.get(
        "/v1/release-readiness",
        response_model=ReleaseReadinessResponse,
        tags=["V1 Governance"],
    )
    def release_readiness(request: Request) -> ReleaseReadinessResponse:
        """Return evidence-backed release posture and explicit production blockers."""
        return services.release_readiness.snapshot(
            _request_context(request, "api_v1_release_readiness")
        )

    @app.get(
        "/v1/release-evidence",
        response_model=ReleaseEvidenceResponse,
        tags=["V1 Governance"],
    )
    def release_evidence(request: Request) -> ReleaseEvidenceResponse:
        """Return hash-addressed package evidence for release review and handoff."""
        return services.release_evidence.snapshot(
            _request_context(request, "api_v1_release_evidence")
        )

    @app.get(
        "/v1/scenarios",
        response_model=list[ScenarioDefinition],
        tags=["V1 Scenarios"],
    )
    def list_scenarios() -> list[ScenarioDefinition]:
        return services.scenarios.list_scenarios()

    @app.get(
        "/v1/scenarios/{scenario_id}",
        response_model=ScenarioDefinition,
        tags=["V1 Scenarios"],
    )
    def get_scenario(scenario_id: str) -> ScenarioDefinition:
        try:
            return services.scenarios.get_scenario(scenario_id)
        except Exception as exc:
            _translate_service_exception(exc)
            raise AssertionError("unreachable") from exc

    @app.post(
        "/v1/scenarios/evaluate",
        response_model=ScenarioEvaluationResponse,
        tags=["V1 Scenarios"],
    )
    def evaluate_scenario(
        request: ScenarioEvaluationRequest,
        http_request: Request,
    ) -> ScenarioEvaluationResponse:
        request = _propagate_correlation(request, http_request)
        try:
            return services.scenarios.evaluate(request)
        except Exception as exc:
            _translate_service_exception(exc)
            raise AssertionError("unreachable") from exc

    @app.get(
        "/v1/decision-packets/{run_id}",
        response_model=DecisionPacketResponse,
        tags=["V1 Governance"],
    )
    def decision_packet(run_id: str, http_request: Request) -> DecisionPacketResponse:
        """Return a portable evidence packet for human review; never authorize execution."""
        try:
            return services.decision_packets.build(
                run_id,
                _request_context(http_request, "api_v1_decision_packet"),
            )
        except DecisionPacketNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/v1/decisions/recommend",
        response_model=DecisionRecommendationResponse,
        tags=["V1 Decisions"],
    )
    def recommend_v1(
        request: DecisionRecommendationRequest,
        http_request: Request,
    ) -> DecisionRecommendationResponse:
        request = _propagate_correlation(request, http_request)
        try:
            return services.decisions.recommend(request)
        except Exception as exc:
            _translate_service_exception(exc)
            raise AssertionError("unreachable") from exc

    @app.post(
        "/v1/audit/{run_id}/review",
        response_model=DecisionReviewResponse,
        tags=["V1 Governance"],
    )
    def review_decision(
        run_id: str,
        request: DecisionReviewRequest,
    ) -> DecisionReviewResponse:
        """Append a human disposition without creating an execution pathway."""
        try:
            review = log_decision_review(
                run_id=run_id,
                reviewed_by=request.reviewed_by,
                decision=request.decision,
                comment=request.comment,
                db_path=services.audit_db_path,
            )
        except AuditRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DecisionReviewResponse(**review)

    @app.get(
        "/v1/audit/reviews",
        response_model=list[DecisionReviewResponse],
        tags=["V1 Governance"],
    )
    def recent_decision_reviews(
        n: int = Query(default=20, ge=1, le=500),
    ) -> list[DecisionReviewResponse]:
        return [
            DecisionReviewResponse(**item)
            for item in get_recent_decision_reviews(n=n, db_path=services.audit_db_path)
        ]

    @app.get(
        "/v1/audit/integrity",
        response_model=AuditIntegrityResponse,
        tags=["V1 Governance"],
    )
    def audit_integrity() -> AuditIntegrityResponse:
        """Verify the tamper-evident human-review chain."""
        return AuditIntegrityResponse(
            **verify_decision_review_chain(db_path=services.audit_db_path)
        )

    @app.get(
        "/v1/audit/{run_id}/reviews",
        response_model=list[DecisionReviewResponse],
        tags=["V1 Governance"],
    )
    def decision_reviews_for_run(run_id: str) -> list[DecisionReviewResponse]:
        if get_decision_run(run_id, db_path=services.audit_db_path) is None:
            raise HTTPException(status_code=404, detail=f"Unknown decision run: {run_id}")
        return [
            DecisionReviewResponse(**item)
            for item in get_reviews_for_run(run_id, db_path=services.audit_db_path)
        ]

    @app.post(
        "/v1/surge-plan",
        response_model=SurgePlanResponse,
        tags=["V1 Surge Planning"],
    )
    def surge_plan(request: SurgePlanRequest, http_request: Request) -> SurgePlanResponse:
        """Run the integrated AI -> IE -> OR -> simulation surge workflow."""
        request = _propagate_correlation(request, http_request)
        try:
            return services.surge_planning.plan(request)
        except Exception as exc:
            _translate_service_exception(exc)
            raise AssertionError("unreachable") from exc

    @app.get("/v1/flow-cvar/reference", tags=["V1 FLOW-CVaR"])
    def flow_cvar_reference() -> dict[str, Any]:
        return build_flow_cvar_decision()

    @app.post("/v1/flow-cvar/decision", tags=["V1 FLOW-CVaR"])
    def flow_cvar_decision(
        cvar_alpha: float = Query(default=0.80, gt=0.0, lt=1.0),
        cvar_weight: float = Query(default=1.50, ge=0.0, le=20.0),
        severe_demand: float = Query(default=22.0, ge=0.0, le=100.0),
    ) -> dict[str, Any]:
        return build_flow_cvar_decision(
            cvar_alpha=cvar_alpha,
            cvar_weight=cvar_weight,
            severe_demand=severe_demand,
        )

    @app.get("/platform-integration-status", tags=["Validation"])
    def platform_integration_status() -> dict[str, Any]:
        if not LEVEL3_WP1_REPORT_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "No Level 3 WP1 report exists. Run "
                    "python -m src.system.run_level3_wp1_validation first."
                ),
            )
        return json.loads(LEVEL3_WP1_REPORT_PATH.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Existing read and validation endpoints
    # ------------------------------------------------------------------
    @app.get("/current-status", tags=["Status"])
    def current_status() -> dict[str, Any]:
        if not COMMAND_CENTER_STATUS_PATH.exists():
            raise HTTPException(status_code=404, detail="No saved command-center status exists.")
        return json.loads(COMMAND_CENTER_STATUS_PATH.read_text(encoding="utf-8"))

    @app.get("/model-governance", tags=["Models"])
    def model_governance() -> dict[str, Any]:
        if not PREDICTIVE_GOVERNANCE_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "No current demand-forecast governance report exists. Run "
                    "python -m src.system.run_finalization_phase1_validation first."
                ),
            )
        payload = json.loads(PREDICTIVE_GOVERNANCE_PATH.read_text(encoding="utf-8"))
        payload.setdefault(
            "governance_note",
            "Current operational forecast evidence uses a genuine chronological date axis. "
            "Earlier replication-index forecasting is retained only as historical research evidence.",
        )
        return payload

    @app.get("/validation-status", tags=["Validation"])
    def validation_status() -> dict[str, Any]:
        if not LEVEL2_COMPLETION_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "No Level 2 completion report exists. Run "
                    "python -m src.system.run_level2_completion_validation first."
                ),
            )
        payload = json.loads(LEVEL2_COMPLETION_PATH.read_text(encoding="utf-8"))
        payload["finalization_note"] = (
            "This Level 2 gate is retained as historical engineering evidence. The finalization audit "
            "supersedes its replication-index forecasting claim and introduced explicit transfer-lane capacity."
        )
        return payload

    @app.get("/finalization-status", tags=["Validation"])
    def finalization_status() -> dict[str, Any]:
        if not FINALIZATION_PHASE1_REPORT_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "No Finalization Phase 1 report exists. Run "
                    "python -m src.system.run_finalization_phase1_validation first."
                ),
            )
        return json.loads(FINALIZATION_PHASE1_REPORT_PATH.read_text(encoding="utf-8"))

    @app.get("/evidence-catalog", tags=["Validation"])
    def evidence_catalog() -> dict[str, Any]:
        if not DATA_ASSET_REGISTRY_SUMMARY_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "No data asset registry exists. Run "
                    "python -m src.system.run_level2_completion_validation first."
                ),
            )
        return json.loads(DATA_ASSET_REGISTRY_SUMMARY_PATH.read_text(encoding="utf-8"))

    @app.get("/alerts", tags=["Alerts"])
    def alerts(n: int = Query(default=20, ge=1, le=500)) -> list[dict[str, Any]]:
        return get_recent_alerts(n=n)

    @app.get("/audit-log", tags=["Audit"])
    def audit_log(n: int = Query(default=10, ge=1, le=500)) -> list[dict[str, Any]]:
        return get_recent_audit_runs(n=n)

    # ------------------------------------------------------------------
    # Backward-compatible aliases, now delegated to the service layer
    # ------------------------------------------------------------------
    @app.post("/what-if", tags=["Decisions"])
    def what_if(request: WhatIfRequest, http_request: Request) -> dict[str, Any]:
        service_request = ScenarioEvaluationRequest(
            scenario_id="custom",
            overrides=ScenarioParameters(
                icu_bed_delta=request.icu_bed_delta,
                transfer_capacity_multiplier=request.transfer_capacity_multiplier,
                demand_surge_multiplier=request.demand_surge_multiplier,
            ),
            n_replications=request.n_replications,
            context=_request_context(http_request, request.surface),
        )
        try:
            return _legacy_whatif_payload(services.scenarios.evaluate(service_request))
        except Exception as exc:
            _translate_service_exception(exc)
            raise AssertionError("unreachable") from exc

    @app.post("/event", tags=["Events"])
    def event(request: EventRequest, http_request: Request) -> dict[str, Any]:
        service_request = ScenarioEvaluationRequest(
            scenario_id=request.event_name,
            n_replications=request.n_replications,
            context=_request_context(http_request, "event_simulator"),
        )
        try:
            response = services.scenarios.evaluate(service_request)
        except Exception as exc:
            _translate_service_exception(exc)
            raise AssertionError("unreachable") from exc
        return {"event_name": request.event_name, **_legacy_whatif_payload(response)}

    @app.post("/recommend", tags=["Decisions"])
    def recommend(request: RecommendRequest, http_request: Request) -> dict[str, Any]:
        service_request = DecisionRecommendationRequest(
            scenario_id="baseline",
            scenario_label=request.scenario_label,
            n_replications=request.n_replications,
            context=_request_context(http_request, "api_legacy_recommend"),
        )
        try:
            return _legacy_recommendation_payload(services.decisions.recommend(service_request))
        except Exception as exc:
            _translate_service_exception(exc)
            raise AssertionError("unreachable") from exc

    return app


app = create_app()
