"""Finalization Phase 1 validation gate for MINCO's mathematical hospital engine."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.benchmarks.stochastic_capacity import make_network_capacity_benchmark
from src.config.paths import PROJECT_ROOT, RESULTS_DIR
from src.decision_math.progressive_hedging import progressive_hedging_single_hospital
from src.decision_math.rolling_horizon import rolling_horizon_reference
from src.decision_math.stochastic_milp_oracle import (
    StochasticCapacityInstance,
    brute_force_single_hospital_first_stage,
    make_tiny_oracle_instance,
    solve_small_stochastic_milp,
    validate_stochastic_solution,
)
from src.ml.discharge_hazard import DiscreteTimeHazardModel, synthetic_stay_dataset
from src.ml.icu_escalation import ICUEscalationModel, evaluate_escalation_model, synthetic_escalation_dataset
from src.queueing.hospital_queueing import kingman_gi_g_1_wait, mm_s_metrics
from src.stochastic.ctmc import make_reference_patient_flow_ctmc, simulate_population_counts
from src.stochastic.decision_scenarios import generate_icu_census_scenarios
from src.stochastic.mmpp import fit_poisson_hmm, posterior_regime_probabilities, simulate_mmpp
from src.stochastic.monte_carlo import (
    CapacityPolicy,
    MonteCarloConfig,
    evaluate_policies_common_random_numbers,
    paired_policy_difference,
)
from src.validation.run_manifest import new_run_id, write_run_manifest
from src.version import __version__

VALIDATION_DIR = RESULTS_DIR / "validation" / "phase1_mathematical_engine"
REPORT_PATH = VALIDATION_DIR / "phase1_mathematical_engine_report.json"
MMPP_PATH = VALIDATION_DIR / "mmpp_regime_validation.json"
HAZARD_PATH = VALIDATION_DIR / "discharge_hazard_validation.json"
ESCALATION_PATH = VALIDATION_DIR / "icu_escalation_validation.json"
MONTE_CARLO_PATH = VALIDATION_DIR / "monte_carlo_policy_summary.csv"
OR_PATH = VALIDATION_DIR / "stochastic_or_validation.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _julia_runtime_probe() -> dict[str, Any]:
    exe = shutil.which("julia")
    if not exe:
        return {
            "available": False,
            "version": None,
            "primary_or_runtime_validated": False,
            "reason": "Julia executable is not installed in this validation environment.",
        }
    try:
        completed = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10, check=True)
        version = completed.stdout.strip() or completed.stderr.strip()
    except Exception as exc:  # pragma: no cover - environment-specific
        return {
            "available": True,
            "version": None,
            "primary_or_runtime_validated": False,
            "reason": f"Julia runtime probe failed: {type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "version": version,
        "primary_or_runtime_validated": False,
        "reason": "Julia exists, but JuMP/Gurobi execution is a licensed Windows acceptance gate.",
    }


def run_validation() -> dict[str, Any]:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id("math_phase1")
    started = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Hidden Markov / MMPP demand-regime inference
    # ------------------------------------------------------------------
    true_transition = np.array([
        [0.92, 0.07, 0.01],
        [0.08, 0.84, 0.08],
        [0.02, 0.12, 0.86],
    ])
    true_rates = np.array([1.5, 3.2, 6.0])
    true_states, counts = simulate_mmpp(
        transition_matrix=true_transition,
        rates=true_rates,
        n_periods=650,
        seed=20260817,
    )
    hmm = fit_poisson_hmm(counts, n_states=3, max_iter=45, tolerance=1e-4, random_state=20260817)
    posterior = posterior_regime_probabilities(counts, hmm)
    inferred_states = np.argmax(posterior, axis=1)
    regime_accuracy = float((inferred_states == true_states).mean())
    rate_relative_error = np.abs(hmm.rates - true_rates) / true_rates
    mmpp_metrics = {
        "true_rates": true_rates.tolist(),
        "estimated_rates": hmm.rates.tolist(),
        "median_rate_relative_error": float(np.median(rate_relative_error)),
        "maximum_rate_relative_error": float(np.max(rate_relative_error)),
        "posterior_regime_accuracy_on_synthetic_truth": regime_accuracy,
        "log_likelihood_initial": float(hmm.log_likelihood_history[0]),
        "log_likelihood_final": float(hmm.log_likelihood_history[-1]),
        "em_iterations": len(hmm.log_likelihood_history),
        "evidence_label": "VALIDATED_ON_SYNTHETIC_GENERATIVE_TRUTH",
    }
    MMPP_PATH.write_text(json.dumps(mmpp_metrics, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 2. CTMC + IE queueing / conservation mathematics
    # ------------------------------------------------------------------
    ctmc = make_reference_patient_flow_ctmc()
    transition_6h = ctmc.transition_matrix(6.0)
    expected_times = ctmc.expected_transient_time("ED")
    absorption = ctmc.absorption_probabilities("ED")
    ctmc_trajectory = simulate_population_counts(
        ctmc,
        initial_counts=[5, 12, 4, 3, 0, 0],
        arrivals_by_step=[3, 4, 2, 5, 3, 4, 2, 4, 3, 5, 2, 4],
        seed=91,
    )
    mass_expected = 24 + np.cumsum([0] + [3, 4, 2, 5, 3, 4, 2, 4, 3, 5, 2, 4])
    mass_error = float(np.max(np.abs(ctmc_trajectory.sum(axis=1) - mass_expected)))

    ed_queue = mm_s_metrics(arrival_rate=8.0, service_rate=2.2, servers=5)
    variability_wait = kingman_gi_g_1_wait(
        arrival_rate=0.72,
        service_rate=1.0,
        arrival_scv=1.7,
        service_scv=1.4,
    )
    ie_metrics = {
        "ctmc_transition_row_sum_error_6h": float(np.max(np.abs(transition_6h.sum(axis=1) - 1.0))),
        "ctmc_expected_transient_hours_from_ed": expected_times,
        "ctmc_absorption_probabilities": absorption,
        "population_mass_conservation_max_error": mass_error,
        "ed_mm5_utilization": ed_queue.utilization,
        "ed_erlang_c_probability_wait": ed_queue.probability_wait,
        "ed_expected_queue_wait_hours": ed_queue.expected_wait,
        "kingman_high_variability_wait_hours": variability_wait,
        "evidence_label": "CALCULATED_AND_SYNTHETICALLY_VALIDATED",
    }

    # ------------------------------------------------------------------
    # 3. Operational survival ML for discharge readiness
    # ------------------------------------------------------------------
    train_stays = synthetic_stay_dataset(1400, seed=111)
    test_stays = synthetic_stay_dataset(500, seed=222)
    hazard_model = DiscreteTimeHazardModel(max_horizon_hours=96).fit(train_stays)
    test_features = test_stays[["age", "acuity_score", "unit", "cohort"]].copy()
    predicted = hazard_model.predict_hazard_curve(test_features, [48])
    p48 = predicted.sort_values("patient_row")["discharge_probability"].to_numpy()
    actual48 = ((test_stays["duration_hours"].to_numpy() <= 48.0) & (test_stays["event_observed"].to_numpy() == 1)).astype(int)
    train_actual48 = ((train_stays["duration_hours"].to_numpy() <= 48.0) & (train_stays["event_observed"].to_numpy() == 1)).astype(int)
    baseline_probability = float(train_actual48.mean())
    model_brier = float(brier_score_loss(actual48, p48))
    baseline_brier = float(brier_score_loss(actual48, np.full_like(p48, baseline_probability, dtype=float)))
    hazard_metrics = {
        "test_patients": int(len(test_stays)),
        "horizon_hours": 48,
        "brier_score": model_brier,
        "constant_baseline_brier_score": baseline_brier,
        "brier_improvement_fraction": float((baseline_brier - model_brier) / baseline_brier),
        "mean_predicted_discharge_probability": float(p48.mean()),
        "observed_synthetic_discharge_fraction": float(actual48.mean()),
        "evidence_label": "VALIDATED_ON_SYNTHETIC_PATIENT_STAYS",
    }
    HAZARD_PATH.write_text(json.dumps(hazard_metrics, indent=2), encoding="utf-8")

    # Calibrated ICU-escalation ML. This is an operational capacity parameter,
    # not a clinical diagnosis or treatment recommendation.
    escalation_train = synthetic_escalation_dataset(2400, seed=333)
    escalation_test = synthetic_escalation_dataset(900, seed=444)
    escalation_model = ICUEscalationModel().fit(escalation_train)
    escalation_metrics = evaluate_escalation_model(escalation_model, escalation_test)
    escalation_metrics["evidence_label"] = "VALIDATED_ON_SYNTHETIC_PATIENT_FEATURES"
    ESCALATION_PATH.write_text(json.dumps(escalation_metrics, indent=2), encoding="utf-8")

    # AI -> OR bridge: MMPP regime inference + discharge hazard + ICU-escalation
    # probabilities directly parameterize stochastic ICU-census scenarios.
    current_icu = test_stays.iloc[:90][["age", "acuity_score", "unit", "cohort"]].copy()
    current_icu["unit"] = "ICU"
    release_curve = hazard_model.predict_hazard_curve(current_icu, [6]).sort_values("patient_row")
    release_probability = np.array([
        float(release_curve.iloc[i*30:(i+1)*30]["discharge_probability"].mean())
        for i in range(3)
    ])
    pending = escalation_test.iloc[:300].copy()
    pending_probability_all = escalation_model.predict_probability(pending)
    escalation_probability = np.array([
        float(pending_probability_all[i*100:(i+1)*100].mean())
        for i in range(3)
    ])
    ai_demand_tensor = generate_icu_census_scenarios(
        hmm_fit=hmm,
        recent_arrival_counts=counts[-72:],
        hospital_arrival_shares=np.array([0.40, 0.35, 0.25]),
        opening_icu_census=np.array([20, 16, 13]),
        discharge_probability_per_period=release_probability,
        escalation_probability=escalation_probability,
        n_periods=4,
        n_scenarios=30,
        period_hours=6.0,
        seed=505,
    )
    ai_or_instance = StochasticCapacityInstance(
        hospitals=("H1", "H2", "H3"),
        periods=(0, 1, 2, 3),
        scenario_probabilities=np.full(30, 1.0 / 30.0),
        demand=ai_demand_tensor,
        base_beds=np.array([24.0, 20.0, 18.0]),
        safe_fraction=np.array([0.90, 0.90, 0.88]),
        surge_beds=np.array([6.0, 5.0, 5.0]),
        surge_cost=np.array([8.0, 7.0, 7.0]),
        flex_beds_per_block=np.array([2.0, 2.0, 2.0]),
        flex_cost=np.array([5.0, 5.0, 5.0]),
        max_flex_blocks=np.array([2, 2, 2]),
        elective_deferral_limit=np.zeros((3, 4)),
        transfer_capacity=np.array([
            [[0,0,0,0],[3,3,3,3],[3,3,3,3]],
            [[3,3,3,3],[0,0,0,0],[3,3,3,3]],
            [[3,3,3,3],[3,3,3,3],[0,0,0,0]],
        ], dtype=float),
        cvar_alpha=0.90,
        cvar_weight=0.50,
    )
    ai_or_solution = solve_small_stochastic_milp(ai_or_instance)
    ai_or_feasibility = validate_stochastic_solution(ai_or_instance, ai_or_solution)
    ai_or_metrics = {
        "scenario_tensor_shape": list(ai_demand_tensor.shape),
        "mean_release_probability_6h": release_probability.tolist(),
        "mean_icu_escalation_probability_24h": escalation_probability.tolist(),
        "solver_status": ai_or_solution.status,
        "objective": ai_or_solution.objective,
        "surge_first_stage": ai_or_solution.surge.tolist(),
        "flex_first_stage": ai_or_solution.flex_blocks.tolist(),
        "feasibility": ai_or_feasibility,
        "decision_chain": "MMPP + discharge hazard + escalation ML -> stochastic ICU census -> CVaR MILP",
        "evidence_label": "SYNTHETIC_AI_PARAMETERIZED_OPTIMIZED_DECISION",
    }

    # ------------------------------------------------------------------
    # 4. Common-random-number Monte Carlo policy experiment
    # ------------------------------------------------------------------
    policies = [
        CapacityPolicy("current_policy"),
        CapacityPolicy(
            "surge_response",
            icu_surge_beds=8,
            ward_surge_beds=16,
            flex_staff_beds=3,
            transfer_diversion_fraction=0.20,
            elective_reduction_fraction=0.15,
        ),
    ]
    mc_detail, mc_summary = evaluate_policies_common_random_numbers(
        policies,
        transition_matrix=true_transition,
        regime_rates=np.array([1.5, 3.2, 6.0]),
        config=MonteCarloConfig(horizon_hours=72, n_scenarios=180, seed=777, cvar_alpha=0.95),
        patient_flow_model=ctmc,
    )
    mc_summary.to_csv(MONTE_CARLO_PATH, index=False)
    paired_loss = paired_policy_difference(
        mc_detail,
        baseline_policy="current_policy",
        candidate_policy="surge_response",
        metric="loss",
    )
    mc_by_policy = mc_summary.set_index("policy")
    monte_carlo_metrics = {
        "n_common_random_scenarios": 180,
        "current_expected_loss": float(mc_by_policy.loc["current_policy", "expected_loss"]),
        "surge_expected_loss": float(mc_by_policy.loc["surge_response", "expected_loss"]),
        "current_cvar95_loss": float(mc_by_policy.loc["current_policy", "cvar_loss"]),
        "surge_cvar95_loss": float(mc_by_policy.loc["surge_response", "cvar_loss"]),
        "mean_paired_loss_difference_candidate_minus_baseline": float(paired_loss.mean()),
        "fraction_scenarios_candidate_no_worse": float((paired_loss <= 1e-9).mean()),
        "evidence_label": "SIMULATED_COMMON_RANDOM_NUMBER_EXPERIMENT",
    }

    # ------------------------------------------------------------------
    # 5. Stochastic MILP, CVaR, enumeration oracle, PH, rolling horizon
    # ------------------------------------------------------------------
    tiny = make_tiny_oracle_instance()
    tiny_solution = solve_small_stochastic_milp(tiny)
    enumeration = brute_force_single_hospital_first_stage(tiny)
    tiny_feasibility = validate_stochastic_solution(tiny, tiny_solution)
    ph = progressive_hedging_single_hospital(tiny, rho=20.0, max_iter=80)

    network = make_network_capacity_benchmark(n_hospitals=3, n_periods=3, n_scenarios=6, seed=43)
    network_solution = solve_small_stochastic_milp(network)
    network_feasibility = validate_stochastic_solution(network, network_solution)

    def rolling_factory(epoch: int):
        base = make_tiny_oracle_instance()
        if epoch == 0:
            return base
        return type(base)(**{**base.__dict__, "demand": base.demand + 4.0 * epoch})

    rolling = rolling_horizon_reference([0, 1, 2], rolling_factory)
    or_metrics = {
        "tiny_extensive_status": tiny_solution.status,
        "tiny_extensive_objective": tiny_solution.objective,
        "tiny_enumeration_objective": float(enumeration["objective"]),
        "tiny_objective_absolute_difference": float(abs(tiny_solution.objective - enumeration["objective"])),
        "tiny_feasibility": tiny_feasibility,
        "progressive_hedging_converged": ph.converged,
        "progressive_hedging_iterations": ph.iterations,
        "progressive_hedging_consensus": {"surge": ph.consensus_surge, "flex": ph.consensus_flex},
        "network_3h_3t_6w_status": network_solution.status,
        "network_3h_3t_6w_objective": network_solution.objective,
        "network_feasibility": network_feasibility,
        "rolling_horizon_objectives": [item.objective for item in rolling],
        "primary_solver": "Julia/JuMP/Gurobi",
        "independent_oracle_solver": "SciPy.milp/HiGHS",
        "evidence_label": "INDEPENDENT_SMALL_INSTANCE_OR_VERIFICATION",
    }
    OR_PATH.write_text(json.dumps(or_metrics, indent=2), encoding="utf-8")

    julia_probe = _julia_runtime_probe()
    checks = {
        "mmpp_em_likelihood_improves": mmpp_metrics["log_likelihood_final"] >= mmpp_metrics["log_likelihood_initial"] - 1e-8,
        "mmpp_rate_error_reasonable": mmpp_metrics["median_rate_relative_error"] <= 0.25,
        "mmpp_regime_recovery_reasonable": mmpp_metrics["posterior_regime_accuracy_on_synthetic_truth"] >= 0.55,
        "ctmc_transition_matrix_stochastic": ie_metrics["ctmc_transition_row_sum_error_6h"] <= 1e-10,
        "ctmc_population_mass_conserved": mass_error == 0.0,
        "queue_model_stable": ed_queue.stable and 0.0 < ed_queue.utilization < 1.0,
        "hazard_model_beats_constant_baseline": hazard_metrics["brier_improvement_fraction"] >= 0.03,
        "icu_escalation_model_discriminates_and_calibrates": escalation_metrics["roc_auc"] >= 0.78 and escalation_metrics["brier_score"] < escalation_metrics["constant_brier_score"],
        "ai_outputs_parameterize_feasible_or_scenarios": ai_or_solution.status == "OPTIMAL" and bool(ai_or_feasibility["feasible"]),
        "monte_carlo_candidate_reduces_expected_loss": monte_carlo_metrics["surge_expected_loss"] <= monte_carlo_metrics["current_expected_loss"],
        "monte_carlo_candidate_reduces_cvar95": monte_carlo_metrics["surge_cvar95_loss"] <= monte_carlo_metrics["current_cvar95_loss"],
        "milp_matches_enumeration": or_metrics["tiny_objective_absolute_difference"] <= 1e-6,
        "milp_solution_independently_feasible": bool(tiny_feasibility["feasible"] and network_feasibility["feasible"]),
        "progressive_hedging_converges_to_extensive_first_stage": bool(
            ph.converged
            and ph.consensus_surge == int(tiny_solution.surge[0,0])
            and ph.consensus_flex == int(tiny_solution.flex_blocks[0,0])
        ),
        "rolling_horizon_reoptimizes": bool(rolling[2].objective >= rolling[0].objective - 1e-8),
        "julia_primary_engine_source_present": all(
            (PROJECT_ROOT / path).exists()
            for path in [
                "julia/src/extensive_form.jl",
                "julia/src/progressive_hedging.jl",
                "julia/src/rolling_horizon.jl",
            ]
        ),
    }
    status = "passed" if all(checks.values()) else "failed"

    manifest_path, manifest_hash = write_run_manifest(
        run_id=run_id,
        run_type="phase1_mathematical_hospital_engine_validation",
        parameters={
            "mmpp_periods": 650,
            "monte_carlo_scenarios": 180,
            "hazard_train_patients": 1400,
            "hazard_test_patients": 500,
        },
        input_paths=[PROJECT_ROOT / "julia/src", PROJECT_ROOT / "src/stochastic", PROJECT_ROOT / "src/decision_math"],
        metrics={
            "status": status,
            "checks": checks,
            "mmpp": mmpp_metrics,
            "industrial_engineering": ie_metrics,
            "discharge_hazard": hazard_metrics,
            "icu_escalation": escalation_metrics,
            "ai_to_or_bridge": ai_or_metrics,
            "monte_carlo": monte_carlo_metrics,
            "operations_research": or_metrics,
        },
        model_info={
            "demand_regime_model": "unsupervised_poisson_hidden_markov_model_baum_welch",
            "patient_flow_model": "continuous_time_markov_chain",
            "discharge_model": "discrete_time_logistic_hazard",
            "icu_escalation_model": "calibrated_gradient_boosting_classifier",
            "ai_to_or_bridge": "learned_probabilities_to_stochastic_icu_census_scenarios",
            "monte_carlo": "common_random_number_policy_experiment",
            "primary_optimization": "julia_jump_gurobi_two_stage_stochastic_milp_cvar",
            "independent_oracle": "scipy_highs_milp_plus_enumeration",
            "decomposition": "progressive_hedging",
            "reoptimization": "rolling_horizon",
        },
        notes=[
            "All numerical validation in this gate uses synthetic generative truth or reference benchmark data.",
            "Julia/JuMP/Gurobi is the primary V1 optimization runtime; the clean-build environment validates source contracts and an independent HiGHS oracle because Julia/Gurobi are not installed here.",
            "Real-hospital external validation remains pending.",
        ],
    )

    report = {
        "phase": "finalization_phase_1_mathematical_hospital_engine",
        "version": __version__,
        "run_id": run_id,
        "generated_at": _utc_now(),
        "status": status,
        "checks": checks,
        "mmpp_regime_engine": mmpp_metrics,
        "ctmc_and_ie": ie_metrics,
        "discharge_hazard_ml": hazard_metrics,
        "icu_escalation_ml": escalation_metrics,
        "ai_to_or_bridge": ai_or_metrics,
        "monte_carlo_lab": monte_carlo_metrics,
        "stochastic_or": or_metrics,
        "julia_runtime_gate": julia_probe,
        "claude_layer": {
            "provider": "Anthropic Claude",
            "routing": "deterministic_cost_aware_haiku_vs_sonnet_family",
            "engineering_role": "evidence_interrogation_only",
            "api_key_required_for_phase1_math": False,
        },
        "evidence_boundary": "VALIDATED ON SYNTHETIC GENERATIVE / BENCHMARK DATA; REAL-HOSPITAL EXTERNAL VALIDATION PENDING",
        "external_acceptance_required": [
            "Run Julia/JuMP/Gurobi primary extensive-form stochastic MILP on the licensed Windows laptop.",
            "Run Julia Progressive Hedging and compare first-stage decisions against the small extensive-form benchmark.",
        ],
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": manifest_hash,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_validation()
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
