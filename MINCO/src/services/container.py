"""Composition root for MINCO application services."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.paths import AUDIT_DB_PATH, DATA_DIR, RESULTS_DIR
from src.services.control_tower_service import ControlTowerService
from src.services.decision_packet_service import DecisionPacketService
from src.services.decision_orchestration_service import DecisionOrchestrationService
from src.services.operational_history_service import OperationalHistoryService
from src.services.operational_state_service import OperationalStateService
from src.services.live_ingestion_service import LiveEventIngestionService
from src.services.release_readiness_service import ReleaseReadinessService
from src.services.release_evidence_service import ReleaseEvidenceService
from src.services.scenario_service import ScenarioRunner, ScenarioService
from src.services.surge_planning_service import SurgePlanningService
from src.validation.run_manifest import RUN_MANIFEST_DIR


@dataclass(frozen=True)
class ServiceContainer:
    operational_state: OperationalStateService
    scenarios: ScenarioService
    decisions: DecisionOrchestrationService
    surge_planning: SurgePlanningService
    control_tower: ControlTowerService
    audit_db_path: Path
    operational_history: OperationalHistoryService
    release_readiness: ReleaseReadinessService
    release_evidence: ReleaseEvidenceService
    live_ingestion: LiveEventIngestionService
    decision_packets: DecisionPacketService


def build_service_container(
    *,
    data_dir: Path = DATA_DIR,
    db_path: Path = AUDIT_DB_PATH,
    manifest_dir: Path = RUN_MANIFEST_DIR,
    scenario_runner: ScenarioRunner | None = None,
    results_dir: Path = RESULTS_DIR,
    event_store: Any | None = None,
) -> ServiceContainer:
    operational_history = OperationalHistoryService(
        data_dir=data_dir,
        results_dir=results_dir,
        store=event_store,
    )
    release_readiness = ReleaseReadinessService(results_dir=results_dir)
    release_evidence = ReleaseEvidenceService(results_dir=results_dir)
    live_ingestion = LiveEventIngestionService(operational_history)
    scenario_service = ScenarioService(
        data_dir=data_dir,
        runner=scenario_runner,
        db_path=db_path,
        manifest_dir=manifest_dir,
    )
    return ServiceContainer(
        operational_state=OperationalStateService(data_dir=data_dir),
        scenarios=scenario_service,
        decisions=DecisionOrchestrationService(
            scenario_service,
            db_path=db_path,
            manifest_dir=manifest_dir,
        ),
        surge_planning=SurgePlanningService(
            scenario_service,
            data_dir=data_dir,
            db_path=db_path,
            manifest_dir=manifest_dir,
        ),
        control_tower=ControlTowerService(
            data_dir=data_dir,
            db_path=db_path,
            operational_history=operational_history,
        ),
        audit_db_path=Path(db_path),
        operational_history=operational_history,
        release_readiness=release_readiness,
        release_evidence=release_evidence,
        live_ingestion=live_ingestion,
        decision_packets=DecisionPacketService(
            db_path=db_path,
            release_evidence=release_evidence,
        ),
    )
