"""Write the current release-provenance attestation as a reviewable artifact."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import RESULTS_DIR
from src.contracts.common import ExecutionContext
from src.services.release_evidence_service import ReleaseEvidenceService

OUTPUT_PATH = RESULTS_DIR / "validation" / "release_evidence_attestation.json"


def write_release_evidence(output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    evidence = ReleaseEvidenceService().snapshot(
        ExecutionContext(source="release_evidence_generator")
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "status": evidence.status,
        "output_path": str(output_path),
        "build_fingerprint": evidence.build_fingerprint,
        "artifact_count": len(evidence.artifacts),
    }


if __name__ == "__main__":
    print(write_release_evidence())
