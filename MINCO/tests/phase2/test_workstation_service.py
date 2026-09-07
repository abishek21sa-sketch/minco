from src.data_runtime.memory_store import MemoryEventStore
from src.runtime.contracts import InterventionRequest, PlanRequest, ReviewRequest
from src.runtime.workstation_service import WorkstationService


def test_workstation_service_couples_state_or_des_and_cost_aware_claude_route():
    service = WorkstationService(store=MemoryEventStore())
    status = service.status()
    assert status.event_count > 450
    assert status.primary_or_solver == "Julia/JuMP/Gurobi"
    state = service.network_state()
    assert len(state["hospitals"]) == 3
    plan = service.plan(
        PlanRequest(n_scenarios=6, planning_horizon_periods=3, use_primary_julia_solver=False)
    )
    assert plan["status"] == "OPTIMAL"
    assert plan["human_review_required"] is True
    assert plan["feasibility"]["feasible"] is True
    sim = service.evaluate_intervention(
        InterventionRequest(
            extra_ed_servers=1,
            extra_ward_beds=4,
            extra_icu_beds=2,
            transfer_out_capacity_per_6h=1,
            n_replications=12,
        )
    )
    assert sim["baseline"]["max_patient_conservation_error"] == 0
    assert sim["candidate"]["max_patient_conservation_error"] == 0
    review = service.review(
        ReviewRequest(
            query="Compare the stochastic optimization and Monte Carlo result and challenge the staffing trade-off",
            execute_external_llm=False,
        )
    )
    assert review["executed"] is False
    assert review["route"]["model_family"] == "sonnet"


def test_primary_solver_failure_falls_back_with_explicit_provenance(monkeypatch):
    service = WorkstationService(store=MemoryEventStore())

    def unavailable(_instance):
        raise RuntimeError("academic Julia environment unavailable")

    monkeypatch.setattr("src.runtime.workstation_service.solve_with_julia_gurobi", unavailable)
    plan = service.plan(
        PlanRequest(n_scenarios=4, planning_horizon_periods=2, use_primary_julia_solver=True)
    )

    assert plan["status"] == "OPTIMAL"
    assert plan["solver"] == "SciPy/HiGHS verification oracle"
    assert plan["solver_fallback"] is True
    assert "academic Julia environment unavailable" in plan["solver_fallback_reason"]
    assert plan["solver_license_mode"] == "academic"
