from __future__ import annotations

from pathlib import Path

from src.storage.audit_repository import (
    get_recent_audit_runs,
    get_run_manifest,
    log_run_manifest,
    log_whatif_run,
)


def test_audit_log_exposes_manifest_reference(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    run_id = log_whatif_run(
        icu_bed_delta=0,
        transfer_multiplier=1.0,
        demand_multiplier=1.0,
        n_replications=3,
        result={"total_unsafe_excess": 1.0},
        surface="test",
        db_path=db_path,
    )
    log_run_manifest(
        run_id,
        manifest_path="results/run_manifests/test.json",
        manifest_sha256="abc123",
        db_path=db_path,
    )

    manifest = get_run_manifest(run_id, db_path=db_path)
    assert manifest is not None
    assert manifest["manifest_sha256"] == "abc123"

    audit_rows = get_recent_audit_runs(n=10, db_path=db_path)
    assert audit_rows[0]["run_id"] == run_id
    assert audit_rows[0]["manifest_path"] == "results/run_manifests/test.json"
    assert audit_rows[0]["manifest_sha256"] == "abc123"
