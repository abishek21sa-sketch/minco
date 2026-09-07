from __future__ import annotations

from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import time

from src.data_runtime.memory_store import MemoryEventStore
from src.data_runtime.replay_engine import build_replay_frames
from src.des.patient_flow import DesConfig, DesIntervention, simulate_patient_flow
from src.hospital_events.synthetic_replay import build_reference_event_replay
from src.runtime.contracts import InterventionRequest, PlanRequest, ReviewRequest
from src.runtime.workstation_service import WorkstationService
from src.state_reconstruction.reconstructor import reconstruct_network_state
from src.version import __version__


def run_validation() -> dict:
    started = time.perf_counter()
    events = build_reference_event_replay()
    store = MemoryEventStore(); store.append(events)
    state = reconstruct_network_state(events)
    frames = build_replay_frames(events, step=timedelta(hours=12))
    cfg = DesConfig(horizon_hours=36, arrival_rate_per_hour=4.0, ed_servers=4, seed=20260823)
    baseline = simulate_patient_flow(cfg)
    candidate = simulate_patient_flow(cfg, DesIntervention(extra_ed_servers=2, extra_ward_beds=6, extra_icu_beds=2, transfer_out_capacity_per_6h=2))
    service = WorkstationService(store=store)
    plan = service.plan(PlanRequest(planning_horizon_periods=3, n_scenarios=6, risk_alpha=.9, risk_weight=.35, use_primary_julia_solver=False))
    sim = service.evaluate_intervention(InterventionRequest(extra_ed_servers=1, extra_ward_beds=4, extra_icu_beds=2, transfer_out_capacity_per_6h=1, n_replications=12))
    review = service.review(ReviewRequest(query="Compare the stochastic plan with the Monte Carlo intervention result and identify the key trade-off", execute_external_llm=False))
    workstation_files = [
        Path("workstation/lib/main.dart"), Path("workstation/lib/screens/census_board.dart"),
        Path("workstation/lib/screens/flow_theatre.dart"), Path("workstation/lib/screens/monte_carlo_room.dart"),
        Path("workstation/lib/screens/intervention_composer.dart"), Path("workstation/lib/screens/decision_review.dart"),
        Path("proto/minco_runtime.proto"),
    ]
    checks = {
        "canonical_event_replay_nontrivial": len(events) > 450,
        "state_reconstruction_three_hospitals": len(state.hospitals) == 3,
        "replay_frames_monotone": len(frames) > 10 and all(a.as_of <= b.as_of for a,b in zip(frames, frames[1:])),
        "des_patient_conservation": baseline.patient_conservation_error == 0 and candidate.patient_conservation_error == 0,
        "des_intervention_improves_ed_wait": candidate.mean_ed_wait_hours <= baseline.mean_ed_wait_hours,
        "stochastic_or_plan_optimal": plan["status"] == "OPTIMAL",
        "stochastic_or_plan_feasible": bool(plan.get("feasibility",{}).get("feasible")),
        "workstation_runtime_simulation_conserves_patients": sim["candidate"]["max_patient_conservation_error"] == 0,
        "claude_router_escalates_cross_module_review": review["route"]["model_family"] == "sonnet",
        "native_workstation_sources_present": all(path.exists() for path in workstation_files),
        "grpc_protobuf_contract_present": "service MincoRuntime" in Path("proto/minco_runtime.proto").read_text(encoding="utf-8"),
    }
    env = {
        "grpc_available": importlib.util.find_spec("grpc") is not None,
        "protobuf_available": importlib.util.find_spec("google.protobuf") is not None,
        "duckdb_available": importlib.util.find_spec("duckdb") is not None,
        "polars_available": importlib.util.find_spec("polars") is not None,
    }
    return {
        "phase":"finalization_phase_2_native_hospital_workstation_core",
        "version":__version__,
        "status":"passed" if all(checks.values()) else "failed",
        "checks":checks,
        "reference_event_count":len(events),
        "replay_frame_count":len(frames),
        "des":{
            "baseline_mean_ed_wait_hours":baseline.mean_ed_wait_hours,
            "candidate_mean_ed_wait_hours":candidate.mean_ed_wait_hours,
            "baseline_p95_boarding_hours":baseline.p95_boarding_hours,
            "candidate_p95_boarding_hours":candidate.p95_boarding_hours,
        },
        "or":{
            "solver":plan["solver"],"objective":plan["objective"],"status":plan["status"],
            "risk_alpha":plan["risk_alpha"],"risk_weight":plan["risk_weight"],
        },
        "claude_route":review["route"],
        "environment":env,
        "external_acceptance_required":[
            "Install Phase-2 DuckDB/Polars dependencies and verify real event-lake + Parquet roundtrip.",
            "Run the native Flutter/Dart Windows workstation build.",
            "Run the workstation plan path against the already-validated Julia/JuMP/Gurobi primary solver.",
        ],
        "evidence_boundary":"SYNTHETIC/REPLAY PLATFORM VALIDATION; REAL-HOSPITAL EXTERNAL VALIDATION PENDING",
        "runtime_seconds":time.perf_counter()-started,
    }


def main() -> None:
    report = run_validation()
    print(json.dumps(report, indent=2, default=str))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
