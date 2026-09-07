from __future__ import annotations

from pathlib import Path

from src.config.paths import DATA_DIR
from src.validation.benchmark_calibration_strategy import run_benchmark_calibration_readiness


def test_benchmark_readiness_passes_and_preserves_claim_boundary(tmp_path: Path) -> None:
    report = run_benchmark_calibration_readiness(
        data_dir=DATA_DIR,
        results_dir=tmp_path / "results",
    )
    assert report["status"] == "passed"
    assert report["current_evidence_basis"] == "synthetic_reference_case"
    assert report["external_historical_calibration_status"] == "not_supplied"
    assert report["real_hospital_deployment_approval"] is False
    assert all(report["checks"].values())
