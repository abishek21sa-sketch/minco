from __future__ import annotations

from pathlib import Path

from src.storage.audit_repository import (
    get_recent_audit_runs,
    get_recent_runs,
    get_recent_whatif_runs,
    log_decision_run,
    log_whatif_run,
)


def test_unified_audit_log_returns_whatif_and_decision_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"

    decision_id = log_decision_run(
        scenario="baseline",
        overall_alert="GREEN",
        pipeline_mode="test",
        runtime_seconds=0.25,
        db_path=db_path,
    )

    whatif_id = log_whatif_run(
        icu_bed_delta=2,
        transfer_multiplier=1.1,
        demand_multiplier=1.2,
        n_replications=3,
        result={
            "total_unsafe_excess": 4.5,
            "max_utilization_ratio": 1.05,
            "total_blocked_arrivals": 2.0,
            "solve_runtime_seconds": 0.1,
        },
        surface="test",
        db_path=db_path,
    )

    combined = get_recent_audit_runs(n=10, db_path=db_path)
    by_id = {row["run_id"]: row for row in combined}

    assert set(by_id) == {decision_id, whatif_id}
    assert by_id[decision_id]["record_type"] == "decision"
    assert by_id[whatif_id]["record_type"] == "what_if"
    assert by_id[whatif_id]["icu_bed_delta"] == 2
    assert by_id[whatif_id]["n_replications"] == 3
    assert by_id[whatif_id]["total_unsafe_excess"] == 4.5

    assert [row["run_id"] for row in get_recent_runs(n=10, db_path=db_path)] == [decision_id]
    assert [row["run_id"] for row in get_recent_whatif_runs(n=10, db_path=db_path)] == [whatif_id]
