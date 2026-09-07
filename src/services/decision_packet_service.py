"""Build portable evidence packets for human-gated decision review."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.contracts.common import ExecutionContext
from src.contracts.decision_packet import DecisionPacketResponse
from src.contracts.release_evidence import ReleaseEvidenceResponse
from src.contracts.review import AuditIntegrityResponse, DecisionReviewResponse
from src.services.release_evidence_service import ReleaseEvidenceService
from src.storage.audit_repository import (
    get_decision_run,
    get_reviews_for_run,
    get_run_manifest,
    get_whatif_run,
    verify_decision_review_chain,
)


class DecisionPacketNotFoundError(ValueError):
    """Raised when a packet is requested for an unknown run."""


def _read_manifest(manifest_reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest_reference:
        return None
    manifest_path = manifest_reference.get("manifest_path")
    if not manifest_path:
        return None
    try:
        payload = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class DecisionPacketService:
    """Compose run, manifest, review, integrity, and release evidence into one packet."""

    service_name = "decision_packet_service"

    def __init__(
        self,
        *,
        db_path: Path,
        release_evidence: ReleaseEvidenceService,
    ) -> None:
        self.db_path = Path(db_path)
        self.release_evidence = release_evidence

    def build(
        self,
        run_id: str,
        context: ExecutionContext,
    ) -> DecisionPacketResponse:
        decision_run = get_decision_run(run_id, db_path=self.db_path)
        if decision_run is None:
            raise DecisionPacketNotFoundError(f"Unknown decision run: {run_id}")

        what_if_run = get_whatif_run(run_id, db_path=self.db_path)
        manifest_reference = get_run_manifest(run_id, db_path=self.db_path)
        manifest = _read_manifest(manifest_reference)
        reviews = [
            DecisionReviewResponse(**item)
            for item in get_reviews_for_run(run_id, db_path=self.db_path)
        ]
        audit_integrity = AuditIntegrityResponse(
            **verify_decision_review_chain(db_path=self.db_path)
        )
        release_evidence: ReleaseEvidenceResponse = self.release_evidence.snapshot(context)

        blockers: list[str] = []
        if manifest_reference is None or manifest is None:
            blockers.append("immutable_run_manifest_unavailable")
        if audit_integrity.status != "VALID":
            blockers.append("audit_integrity_invalid")
        if release_evidence.status != "VERIFIED":
            blockers.append("release_evidence_not_verified")

        latest_review = reviews[-1] if reviews else None
        review_status = latest_review.decision if latest_review else "PENDING"
        return DecisionPacketResponse(
            packet_id=f"packet_{run_id}",
            correlation_id=context.correlation_id,
            run_id=run_id,
            decision_run=decision_run,
            what_if_run=what_if_run,
            manifest_reference=manifest_reference,
            manifest=manifest,
            reviews=reviews,
            audit_integrity=audit_integrity,
            release_evidence=release_evidence,
            governance={
                "packet_ready": not blockers,
                "review_status": review_status,
                "blockers": blockers,
            },
        )
