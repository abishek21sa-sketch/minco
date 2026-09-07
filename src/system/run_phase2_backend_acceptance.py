"""Windows acceptance gate for Phase-2 external runtime dependencies."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from src.data_runtime.duckdb_store import DuckDBEventStore
from src.data_runtime.polars_analytics import summarize_parquet_with_polars
from src.config.paths import RESULTS_DIR
from src.hospital_events.synthetic_replay import build_reference_event_replay
from src.optimization_julia.bridge import check_julia_readiness
from src.runtime.contracts import PlanRequest
from src.runtime.workstation_service import WorkstationService


def run_acceptance() -> dict:
    with tempfile.TemporaryDirectory(prefix="minco_phase2_") as tmp:
        root = Path(tmp)
        store = DuckDBEventStore(root / "events.duckdb")
        events = build_reference_event_replay()
        inserted = store.append(events)
        parquet = store.export_parquet(root / "events.parquet")
        polars = summarize_parquet_with_polars(parquet)
        service = WorkstationService(store=store)
        julia_readiness = check_julia_readiness()
        julia_ready = julia_readiness.ready
        plan = service.plan(PlanRequest(planning_horizon_periods=2,n_scenarios=4,risk_alpha=.9,risk_weight=.35,use_primary_julia_solver=julia_ready))
        checks = {
            "duckdb_roundtrip_complete": store.count == len(events) and inserted == len(events),
            "parquet_written": parquet.exists() and parquet.stat().st_size > 0,
            "polars_reads_all_events": polars["rows"] == len(events),
            "workstation_plan_optimal": plan["status"] == "OPTIMAL",
            "primary_julia_solver_used": (not julia_ready) or plan["solver"] == "Julia/JuMP/Gurobi",
        }
        report = {
            "status":"passed" if all(checks.values()) else "failed",
            "checks":checks,
            "event_count":len(events),
            "parquet_bytes":parquet.stat().st_size if parquet.exists() else 0,
            "polars_groups":len(polars["hospital_event_summary"]),
            "plan_solver":plan["solver"],
            "plan_objective":plan["objective"],
            "julia_available": julia_readiness.executable_available,
            "julia_readiness": julia_readiness.to_dict(),
        }
        report_path = RESULTS_DIR / "validation" / "phase2_backend_acceptance_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return report


def main() -> None:
    report = run_acceptance()
    print(json.dumps(report, indent=2, default=str))
    raise SystemExit(0 if report["status"] == "passed" else 1)

if __name__ == "__main__":
    main()
