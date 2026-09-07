"""Build a bounded, hash-addressed release evidence attestation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config.paths import RESULTS_DIR
from src.contracts.common import ExecutionContext, new_correlation_id, utc_now
from src.contracts.release_evidence import ReleaseEvidenceArtifact, ReleaseEvidenceResponse
from src.validation.run_manifest import sha256_file


class ReleaseEvidenceService:
    """Report reproducibility of the candidate evidence package without overclaiming."""

    service_name = "release_evidence_service"

    def __init__(self, results_dir: Path = RESULTS_DIR) -> None:
        self.results_dir = Path(results_dir)
        self.project_root = self.results_dir.parent

    def _artifact_specs(self) -> list[tuple[str, Path]]:
        return [
            (
                "MINCO/results/contracts/service_contract_registry.json",
                self.results_dir / "contracts" / "service_contract_registry.json",
            ),
            (
                "MINCO/results/validation/phase2_backend_acceptance_report.json",
                self.results_dir / "validation" / "phase2_backend_acceptance_report.json",
            ),
            (
                "MINCO/artifacts/product_runtime/latest_product_evidence.json",
                self.project_root / "artifacts" / "product_runtime" / "latest_product_evidence.json",
            ),
            (
                "MINCO/PRODUCT_V1_RELEASE_MANIFEST.json",
                self.project_root / "PRODUCT_V1_RELEASE_MANIFEST.json",
            ),
            (
                "MINCO/docs/execution/PROJECT_STATUS.md",
                self.project_root / "docs" / "execution" / "PROJECT_STATUS.md",
            ),
        ]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def snapshot(self, context: ExecutionContext | None = None) -> ReleaseEvidenceResponse:
        context = context or ExecutionContext(
            correlation_id=new_correlation_id(), source="service"
        )
        artifacts: list[ReleaseEvidenceArtifact] = []
        present_hashes: list[str] = []
        missing_artifacts: list[str] = []

        for relative_path, path in self._artifact_specs():
            if not path.exists() or not path.is_file():
                missing_artifacts.append(relative_path)
                artifacts.append(
                    ReleaseEvidenceArtifact(
                        path=relative_path,
                        required=True,
                        exists=False,
                        status="MISSING",
                    )
                )
                continue
            try:
                digest = sha256_file(path)
                size_bytes = path.stat().st_size
            except (OSError, ValueError):
                missing_artifacts.append(relative_path)
                artifacts.append(
                    ReleaseEvidenceArtifact(
                        path=relative_path,
                        required=True,
                        exists=True,
                        status="UNHASHABLE",
                    )
                )
                continue
            present_hashes.append(f"{relative_path}={digest}")
            artifacts.append(
                ReleaseEvidenceArtifact(
                    path=relative_path,
                    required=True,
                    exists=True,
                    status="VERIFIED",
                    sha256=digest,
                    size_bytes=size_bytes,
                )
            )

        registry = self._read_json(self.results_dir / "contracts" / "service_contract_registry.json")
        phase2 = self._read_json(
            self.results_dir / "validation" / "phase2_backend_acceptance_report.json"
        )
        runtime_evidence = self._read_json(
            self.project_root / "artifacts" / "product_runtime" / "latest_product_evidence.json"
        )
        manifest = self._read_json(self.project_root / "PRODUCT_V1_RELEASE_MANIFEST.json")
        gates = {
            "contract_registry_present": bool(registry.get("schemas")),
            "phase2_backend_acceptance_passed": phase2.get("status") == "passed",
            "product_runtime_acceptance_evidence_present": bool(runtime_evidence),
            "release_manifest_present": bool(manifest.get("release")),
            "required_artifacts_hashed": not missing_artifacts and len(present_hashes) == len(artifacts),
        }
        build_fingerprint = None
        if present_hashes:
            build_fingerprint = hashlib.sha256(
                "\n".join(sorted(present_hashes)).encode("utf-8")
            ).hexdigest()
        status = "VERIFIED" if all(gates.values()) else "BLOCKED"
        release_id = str(manifest.get("release", "MINCO_PRODUCT_V1_CANDIDATE"))
        if status == "VERIFIED":
            statement = (
                "All required candidate artifacts were present and SHA-256 hashed. "
                "This verifies evidence-package integrity only; it does not establish "
                "clinical, real-hospital, live-feed, enterprise-IAM, or autonomous-use approval."
            )
        else:
            statement = (
                "The candidate evidence package is incomplete or failed a consistency gate. "
                "Do not use this attestation as a release approval."
            )

        return ReleaseEvidenceResponse(
            generated_at=utc_now(),
            correlation_id=context.correlation_id,
            release_id=release_id,
            status=status,
            build_fingerprint=build_fingerprint,
            artifacts=artifacts,
            gates=gates,
            missing_artifacts=missing_artifacts,
            integrity_statement=statement,
        )
