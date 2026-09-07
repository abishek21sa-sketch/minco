"""Application service used by the native workstation gRPC boundary."""
from __future__ import annotations

from dataclasses import asdict
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmarks.stochastic_capacity import make_network_capacity_benchmark
from src.claude.client import ask_claude
from src.claude.router import route_claude_query
from src.data_runtime.memory_store import MemoryEventStore
from src.data_runtime.factory import make_event_store
from src.des.patient_flow import DesConfig, DesIntervention, simulate_patient_flow
from src.hospital_events.synthetic_replay import build_reference_event_replay
from src.runtime.contracts import InterventionRequest, PlanRequest, ReviewRequest, RuntimeStatus
from src.runtime.evidence_tools import build_review_evidence
from src.scenarios.operating_modes import get_operating_mode, operating_mode_catalog
from src.scenarios.public_reference import load_facility_reference
from src.state_reconstruction.reconstructor import reconstruct_network_state
from src.decision_math.stochastic_milp_oracle import solve_small_stochastic_milp, validate_stochastic_solution
from src.optimization_julia.bridge import configured_solver_license_mode, solve_with_julia_gurobi
from src.version import __version__


class WorkstationService:
    """Deterministic service façade. No web framework is required by the product."""

    def __init__(self, store: MemoryEventStore | Any | None = None) -> None:
        if store is None:
            self.store, self.data_stack, self.store_warnings = make_event_store()
        else:
            self.store = store
            self.data_stack = store.__class__.__name__
            self.store_warnings = []
            if getattr(self.store, "count", 0) == 0:
                self.store.append(build_reference_event_replay())
        self._last_plan: dict[str, Any] | None = None
        self._last_simulation: dict[str, Any] | None = None
        self._operating_mode = os.getenv("MINCO_OPERATING_MODE", "normal").strip().lower()
        get_operating_mode(self._operating_mode)
        is_memory = self.store.__class__.__name__ == "MemoryEventStore"
        configured_facilities = int(os.getenv("MINCO_REPLAY_HOSPITALS", "120"))
        facility_count = 3 if is_memory else max(4, configured_facilities)
        reference_path = Path(os.getenv(
            "MINCO_PUBLIC_REFERENCE_CSV",
            str(Path(os.getenv("MINCO_DATA_DIR", "data")) / "public_reference" / "cms_hospital_general_information.csv"),
        ))
        self._facility_reference = load_facility_reference(
            n_hospitals=facility_count,
            csv_path=None if is_memory else reference_path,
        )

    def status(self) -> RuntimeStatus:
        latest = self.store.latest_timestamp()
        return RuntimeStatus(
            version=__version__,
            data_mode="SYNTHETIC_REFERENCE_REPLAY" if self.store.__class__.__name__ == "MemoryEventStore" else "SCALED_SYNTHETIC_EVENT_LAKE",
            event_count=int(self.store.count),
            data_stack=self.data_stack,
            latest_event_timestamp=latest,
            solver_license_mode=configured_solver_license_mode(),
            solver_fallback_enabled=True,
            facility_count=len(self._facility_reference.facilities),
            operating_mode=self._operating_mode,
            available_operating_modes=[str(item["mode"]) for item in operating_mode_catalog()],
            dataset_provenance=self._facility_reference.provenance,
            evidence_boundary="Synthetic/replay validation only; real-hospital external validation pending.",
            warnings=[*([] if latest else ["No events loaded"]), *self.store_warnings],
        )

    def network_state(self) -> dict[str, Any]:
        events = self.store.events_between()
        state = reconstruct_network_state(events)
        payload = state.model_dump(mode="json")
        payload["provenance"] = "OBSERVED_FROM_CANONICAL_REPLAY_EVENTS"
        payload["scale_metadata"] = {
            "facility_count": len(self._facility_reference.facilities),
            "operating_mode": self._operating_mode,
            "available_operating_modes": operating_mode_catalog(),
            "dataset_provenance": self._facility_reference.provenance,
            "capacity_claim": "Synthetic capacities and patient-flow events; public directory metadata is calibration context only.",
        }
        return payload

    def plan(self, request: PlanRequest) -> dict[str, Any]:
        instance = make_network_capacity_benchmark(
            n_hospitals=request.network_hospitals,
            n_periods=request.planning_horizon_periods,
            n_scenarios=min(request.n_scenarios, 80),
            seed=20260823,
            operating_mode=request.operating_mode,
        )
        # Request-level CVaR semantics directly change the OR objective.
        instance = instance.__class__(**{
            **instance.__dict__,
            "cvar_alpha": request.risk_alpha,
            "cvar_weight": request.risk_weight,
        })
        solver = "SciPy/HiGHS verification oracle"
        payload: dict[str, Any] | None = None
        fallback_reason: str | None = None
        if request.use_primary_julia_solver:
            try:
                primary = solve_with_julia_gurobi(instance)
                p = primary.payload
                payload = {
                    "solver": "Julia/JuMP/Gurobi",
                    "status": p.get("termination_status"),
                    "objective": p.get("objective"),
                    "relative_gap": p.get("relative_gap"),
                    "surge": p.get("surge"),
                    "flex_blocks": p.get("flex_blocks"),
                    "cvar": p.get("cvar"),
                    "eta": p.get("eta"),
                    "scenario_loss": p.get("scenario_loss"),
                }
            except Exception as exc:
                fallback_reason = (
                    "Primary Julia/JuMP/Gurobi solver was unavailable; the governed "
                    f"SciPy/HiGHS verification oracle was used instead. {exc}"
                )

        if payload is None:
            solution = solve_small_stochastic_milp(instance)
            feasibility = validate_stochastic_solution(instance, solution)
            payload = {
                "solver": solver,
                "status": solution.status,
                "objective": solution.objective,
                "relative_gap": None,
                "surge": solution.surge.tolist(),
                "flex_blocks": solution.flex_blocks.tolist(),
                "cvar": float(solution.eta + np.average(solution.xi, weights=instance.scenario_probabilities) / (1 - instance.cvar_alpha)),
                "eta": solution.eta,
                "scenario_loss": solution.scenario_loss.tolist(),
                "feasibility": feasibility,
            }
        payload.update({
            "planning_horizon_periods": request.planning_horizon_periods,
            "n_scenarios": min(request.n_scenarios, 80),
            "risk_alpha": request.risk_alpha,
            "risk_weight": request.risk_weight,
            "risk_posture": request.risk_posture,
            "provenance": "OPTIMIZED_SYNTHETIC_DECISION",
            "human_review_required": True,
            "solver_license_mode": configured_solver_license_mode(),
            "primary_solver_requested": bool(request.use_primary_julia_solver),
            "solver_fallback": fallback_reason is not None,
            "solver_fallback_reason": fallback_reason,
            "operating_mode": request.operating_mode,
            "network_hospitals": request.network_hospitals,
            "operating_mode_profile": get_operating_mode(request.operating_mode).__dict__,
            "claim_boundary": "Optimized synthetic network decision support; no autonomous or clinical execution.",
        })
        self._last_plan = payload
        return payload

    def evaluate_intervention(self, request: InterventionRequest) -> dict[str, Any]:
        mode = get_operating_mode(request.operating_mode)
        baseline_results = []
        candidate_results = []
        for rep in range(request.n_replications):
            seed = 20260823 + rep
            base_cfg = DesConfig(
                seed=seed,
                arrival_rate_per_hour=3.0 * mode.demand_multiplier,
                icu_given_admission_probability=min(0.95, 0.18 * mode.icu_multiplier),
                ward_los_mean_hours=30.0 * mode.length_of_stay_multiplier,
                icu_los_mean_hours=42.0 * mode.length_of_stay_multiplier,
            )
            baseline_results.append(simulate_patient_flow(base_cfg))
            candidate_results.append(simulate_patient_flow(base_cfg, DesIntervention(
                extra_ward_beds=request.extra_ward_beds,
                extra_icu_beds=request.extra_icu_beds,
                extra_ed_servers=request.extra_ed_servers,
                transfer_out_capacity_per_6h=request.transfer_out_capacity_per_6h,
            )))

        def summarize(results):
            return {
                "mean_ed_wait_hours": float(np.mean([r.mean_ed_wait_hours for r in results])),
                "p95_ed_wait_hours": float(np.quantile([r.mean_ed_wait_hours for r in results], 0.95)),
                "mean_boarding_hours": float(np.mean([r.mean_boarding_hours for r in results])),
                "p95_boarding_hours": float(np.quantile([r.mean_boarding_hours for r in results], 0.95)),
                "mean_max_icu_occupancy": float(np.mean([r.max_icu_occupancy for r in results])),
                "mean_max_ward_occupancy": float(np.mean([r.max_ward_occupancy for r in results])),
                "mean_transfers_out": float(np.mean([r.transferred_out for r in results])),
                "max_patient_conservation_error": int(max(abs(r.patient_conservation_error) for r in results)),
            }
        baseline = summarize(baseline_results)
        candidate = summarize(candidate_results)
        result = {
            "n_common_random_replications": request.n_replications,
            "baseline": baseline,
            "candidate": candidate,
            "paired_delta": {
                "mean_ed_wait_hours": candidate["mean_ed_wait_hours"] - baseline["mean_ed_wait_hours"],
                "mean_boarding_hours": candidate["mean_boarding_hours"] - baseline["mean_boarding_hours"],
            },
            "provenance": "SIMULATED_COMMON_RANDOM_NUMBER_DES_EXPERIMENT",
            "claim_boundary": "Modeled operational comparison only; not a realized clinical benefit.",
            "operating_mode": mode.mode,
            "operating_mode_profile": mode.__dict__,
        }
        self._last_simulation = result
        return result

    def scenario_catalog(self) -> dict[str, Any]:
        return {
            "operating_modes": operating_mode_catalog(),
            "active_mode": self._operating_mode,
            "facility_count": len(self._facility_reference.facilities),
            "dataset_provenance": self._facility_reference.provenance,
            "claim_boundary": "Modes are synthetic stress-test parameterizations, not clinical disease classifiers.",
        }

    def review(self, request: ReviewRequest) -> dict[str, Any]:
        route = route_claude_query(
            request.query,
            tool_count=len(request.evidence_scope),
            requires_cross_module_reasoning=len(request.evidence_scope) >= 2,
        )
        state = self.network_state() if "state" in request.evidence_scope else None
        plan = self._last_plan if "plan" in request.evidence_scope else None
        simulation = self._last_simulation if "simulation" in request.evidence_scope else None
        evidence = build_review_evidence(state=state, plan=plan, simulation=simulation)
        if not request.execute_external_llm:
            return {
                "provider": "Anthropic Claude",
                "route": route.__dict__,
                "executed": False,
                "evidence_bytes": len(evidence.encode("utf-8")),
                "message": "Routing and deterministic evidence prepared. External Claude call disabled.",
            }
        response = ask_claude(
            request.query,
            system_prompt=(
                "You are MINCO Decision Review Board. Use only supplied deterministic evidence. "
                "Never turn simulated or optimized values into observed outcomes. Identify binding assumptions and trade-offs."
            ),
            evidence=evidence,
            tool_count=len(request.evidence_scope),
            requires_cross_module_reasoning=len(request.evidence_scope) >= 2,
        )
        return {
            "provider": "Anthropic Claude",
            "model": response.model,
            "route": response.route.__dict__,
            "executed": True,
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
