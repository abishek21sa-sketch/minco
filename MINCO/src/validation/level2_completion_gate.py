"""Evidence-integrity gate that closes MINCO Level 2."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from src.config.paths import RESULTS_DIR
from src.validation.benchmark_calibration_strategy import run_benchmark_calibration_readiness
from src.validation.data_asset_registry import build_data_asset_registry
from src.validation.run_manifest import new_run_id, sha256_file, write_run_manifest
from src.version import __version__

REPORT_NAMES = [
    "level2_wp1_validation_report.json",
    "level2_wp2_validation_report.json",
    "level2_wp3_validation_report.json",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_manifest_references(value: Any) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    if isinstance(value, dict):
        path = value.get("manifest_path")
        digest = value.get("manifest_sha256")
        if path and digest:
            references.append((str(path), str(digest)))
        for nested in value.values():
            references.extend(_collect_manifest_references(nested))
    elif isinstance(value, list):
        for nested in value:
            references.extend(_collect_manifest_references(nested))
    return references


def _resolve_manifest(path_text: str, manifest_dir: Path) -> Path:
    candidate = Path(path_text)
    if candidate.exists():
        return candidate
    name = PureWindowsPath(path_text).name if "\\" in path_text else candidate.name
    return manifest_dir / name


def verify_manifest_references(
    reports: list[dict[str, Any]], *, manifest_dir: Path
) -> dict[str, object]:
    unique = sorted(set(ref for report in reports for ref in _collect_manifest_references(report)))
    details: list[dict[str, object]] = []
    for path_text, expected_hash in unique:
        resolved = _resolve_manifest(path_text, manifest_dir)
        exists = resolved.exists()
        observed_hash = sha256_file(resolved) if exists else None
        details.append(
            {
                "path": str(resolved),
                "exists": exists,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "valid": bool(exists and observed_hash == expected_hash),
            }
        )
    return {
        "reference_count": len(details),
        "valid_count": int(sum(bool(item["valid"]) for item in details)),
        "all_valid": bool(details and all(bool(item["valid"]) for item in details)),
        "details": details,
    }


def run_level2_completion_gate(
    *,
    results_dir: Path = RESULTS_DIR,
    manifest_dir: Path | None = None,
) -> dict[str, object]:
    validation_dir = results_dir / "validation"
    manifest_dir = Path(manifest_dir or (results_dir / "run_manifests"))
    report_paths = [validation_dir / name for name in REPORT_NAMES]
    missing_reports = [str(path) for path in report_paths if not path.exists()]
    reports = [_load_json(path) for path in report_paths if path.exists()]
    work_package_status = {
        path.stem: report.get("status")
        for path, report in zip(
            [path for path in report_paths if path.exists()], reports, strict=True
        )
    }

    benchmark = run_benchmark_calibration_readiness(
        results_dir=results_dir, manifest_dir=manifest_dir
    )
    registry = build_data_asset_registry(
        results_dir=results_dir, manifest_dir=manifest_dir
    )
    evidence_sources = [*reports, benchmark, registry]
    manifest_integrity = verify_manifest_references(
        evidence_sources, manifest_dir=manifest_dir
    )

    wp3 = next(
        (report for report in reports if report.get("work_package", "").startswith("level_2_wp3")),
        {},
    )
    decisions = wp3.get("predictive_model_governance", {}).get("task_decisions", [])
    approved_models = [
        item for item in decisions if item.get("status") == "approved_for_reference_case"
    ]
    governed_task_count = len(decisions)

    checks = {
        "all_work_package_reports_present": not missing_reports and len(reports) == 3,
        "all_work_packages_passed": bool(reports)
        and all(report.get("status") == "passed" for report in reports),
        "benchmark_strategy_passed": benchmark.get("status") == "passed",
        "asset_registry_passed": registry.get("status") == "passed",
        "manifest_integrity_passed": manifest_integrity["all_valid"],
        "predictive_tasks_governed": governed_task_count == 4,
        "approved_models_are_reference_case_only": bool(approved_models)
        and all(item.get("status") == "approved_for_reference_case" for item in approved_models),
        "external_calibration_boundary_explicit": benchmark.get(
            "external_historical_calibration_status"
        )
        == "not_supplied",
    }
    status = "passed" if all(checks.values()) else "failed"
    run_id = new_run_id("level2_completion")
    report: dict[str, object] = {
        "work_package": "level_2_wp4_benchmark_registry_and_completion_gate",
        "version": __version__,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "level_2_disposition": (
            "complete_and_level_3_authorized" if status == "passed" else "not_complete"
        ),
        "work_package_status": work_package_status,
        "missing_reports": missing_reports,
        "checks": checks,
        "benchmark_calibration_readiness": benchmark,
        "data_asset_registry": registry,
        "manifest_integrity": manifest_integrity,
        "predictive_governance_summary": {
            "governed_task_count": governed_task_count,
            "approved_reference_case_model_count": len(approved_models),
            "approved_tasks": [item.get("task_name") for item in approved_models],
        },
        "claim_boundary": (
            "Level 2 completion establishes reproducible analytical validity for the bundled "
            "synthetic reference case. It does not establish real-hospital or clinical validity."
        ),
        "required_follow_up": [] if status == "passed" else [
            key for key, passed in checks.items() if not passed
        ],
    }
    validation_dir.mkdir(parents=True, exist_ok=True)
    report_path = validation_dir / "level2_completion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path, manifest_hash = write_run_manifest(
        run_id=run_id,
        run_type="level2_completion_gate",
        parameters={"required_work_packages": REPORT_NAMES},
        input_paths=report_paths,
        output_paths=[results_dir / "data_registry" / "data_asset_registry.csv"],
        metrics={
            "passed_check_count": int(sum(checks.values())),
            "check_count": len(checks),
            "verified_manifest_count": manifest_integrity["valid_count"],
            "approved_reference_case_model_count": len(approved_models),
        },
        notes=[report["claim_boundary"]],
        manifest_dir=manifest_dir,
    )
    report["manifest_path"] = str(manifest_path)
    report["manifest_sha256"] = manifest_hash
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
