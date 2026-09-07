from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import DATA_DIR
from src.validation.level2_completion_gate import (
    run_level2_completion_gate,
    verify_manifest_references,
)
from src.validation.run_manifest import sha256_file


def _write_manifest(path: Path, run_id: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    return sha256_file(path)


def _write_reports(results_dir: Path) -> None:
    validation = results_dir / "validation"
    manifests = results_dir / "run_manifests"
    validation.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(
        [
            "level2_wp1_validation_report.json",
            "level2_wp2_validation_report.json",
        ],
        start=1,
    ):
        manifest = manifests / f"wp{index}.json"
        digest = _write_manifest(manifest, f"wp{index}")
        (validation / name).write_text(
            json.dumps(
                {"status": "passed", "manifest_path": str(manifest), "manifest_sha256": digest}
            ),
            encoding="utf-8",
        )
    manifest = manifests / "wp3.json"
    digest = _write_manifest(manifest, "wp3")
    (validation / "level2_wp3_validation_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "work_package": "level_2_wp3_predictive_model_validation_and_governance",
                "manifest_path": str(manifest),
                "manifest_sha256": digest,
                "predictive_model_governance": {
                    "task_decisions": [
                        {"task_name": "next_unsafe_excess", "status": "rejected"},
                        {
                            "task_name": "next_blocked_arrivals",
                            "status": "approved_for_reference_case",
                        },
                        {"task_name": "next_utilization_critical", "status": "rejected"},
                        {"task_name": "next_regime", "status": "rejected"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_manifest_integrity_detects_tampering(tmp_path: Path) -> None:
    manifest = tmp_path / "run.json"
    digest = _write_manifest(manifest, "run")
    report = {"manifest_path": str(manifest), "manifest_sha256": digest}
    assert verify_manifest_references([report], manifest_dir=tmp_path)["all_valid"] is True
    manifest.write_text('{"run_id":"tampered"}', encoding="utf-8")
    assert verify_manifest_references([report], manifest_dir=tmp_path)["all_valid"] is False


def test_level2_completion_gate_passes_with_complete_evidence(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    _write_reports(results_dir)
    monkeypatch.setattr("src.validation.level2_completion_gate.RESULTS_DIR", results_dir)
    monkeypatch.setattr("src.validation.benchmark_calibration_strategy.DATA_DIR", DATA_DIR)

    report = run_level2_completion_gate(results_dir=results_dir)

    assert report["status"] == "passed"
    assert report["level_2_disposition"] == "complete_and_level_3_authorized"
    assert report["checks"]["manifest_integrity_passed"] is True
    assert report["predictive_governance_summary"]["approved_tasks"] == ["next_blocked_arrivals"]
