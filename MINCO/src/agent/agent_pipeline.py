from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class AgentStepResult:
    agent_name: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


class ForecastAgent:
    """
    Reads the 7-day Digital Twin horizon (best/expected/worst bottleneck
    trajectory) and reports the forward-looking read for the final day.
    Wraps compute_digital_twin_horizon's output rather than recomputing
    forecasting logic.
    """

    name = "Forecast Agent"

    def run(self, twin_horizon_final_day: Dict[str, Any]) -> AgentStepResult:
        expected = float(twin_horizon_final_day.get("expected", 0.0))
        worst = float(twin_horizon_final_day.get("worst", 0.0))
        day = twin_horizon_final_day.get("day")

        summary = (
            f"7-day forecast (day {day}): expected bottleneck score {expected:.2f}, "
            f"worst-case {worst:.2f}."
        )
        return AgentStepResult(self.name, summary, {"expected": expected, "worst": worst, "day": day})


class OptimizationAgent:
    """
    Wraps the live network optimization solve (Sandbox/Copilot/Event Simulator's
    underlying run_live_whatif) rather than re-solving anything new.
    """

    name = "Optimization Agent"

    def run(self, whatif_fn: Callable[..., Dict[str, float]], whatif_kwargs: Dict[str, Any]) -> AgentStepResult:
        result = whatif_fn(**whatif_kwargs)

        summary = (
            f"Solved optimized_network policy: unsafe excess {result.get('total_unsafe_excess', 0.0):.2f}, "
            f"max utilization {result.get('max_utilization_ratio', 0.0):.3f} "
            f"({result.get('solve_runtime_seconds', 0.0):.2f}s, {int(result.get('n_replications', 0))} reps)."
        )
        return AgentStepResult(self.name, summary, result)


class RiskAgent:
    """
    Classifies risk from the *optimized* outcome (not pre-optimization), using
    the same classify_stress_label thresholds the rest of the dashboard uses,
    so risk language stays consistent everywhere.
    """

    name = "Risk Agent"

    def run(
        self,
        opt_result: AgentStepResult,
        classify_fn: Callable[[float, float, float], tuple],
    ) -> AgentStepResult:
        max_util = float(opt_result.details.get("max_utilization_ratio", 0.0))
        overflow = float(opt_result.details.get("total_overflow_excess", 0.0))
        unsafe_rows = float(opt_result.details.get("num_unsafe_rows", 0.0))

        icu_status, overflow_status, bottleneck = classify_fn(max_util, overflow, unsafe_rows)

        summary = f"ICU status: {icu_status}; overflow risk: {overflow_status}; bottleneck: {bottleneck}."
        return AgentStepResult(
            self.name,
            summary,
            {"icu_status": icu_status, "overflow_status": overflow_status, "bottleneck": bottleneck},
        )


class ExplanationAgent:
    """
    Turns the Risk Agent's classification plus the Optimization Agent's
    outcome into a single executive-readable recommendation.
    """

    name = "Explanation Agent"

    def run(self, risk_result: AgentStepResult, opt_result: AgentStepResult) -> AgentStepResult:
        icu_status = risk_result.details.get("icu_status", "Unknown")
        unsafe = float(opt_result.details.get("total_unsafe_excess", 0.0))

        if icu_status == "Critical":
            recommendation = "Activate surge capacity and prioritize inter-hospital transfers immediately."
        elif icu_status == "Stressed":
            recommendation = "Maintain transfer-enabled coordination; monitor ICU utilization closely."
        else:
            recommendation = "No aggressive intervention required; continue standard coordinated operation."

        summary = f"Recommendation: {recommendation} (current unsafe excess: {unsafe:.2f})"
        return AgentStepResult(self.name, summary, {"recommendation": recommendation})


class ScenarioAgent:
    """
    Tests a capacity-expansion variant against the Optimization Agent's
    baseline run, using the same live solver -- a second real solve, not a
    guess -- and reports whether it would meaningfully reduce unsafe excess.
    """

    name = "Scenario Agent"

    def run(
        self,
        opt_result: AgentStepResult,
        whatif_fn: Callable[..., Dict[str, float]],
        whatif_kwargs: Dict[str, Any],
        icu_bed_variant: int = 15,
    ) -> AgentStepResult:
        variant_kwargs = dict(whatif_kwargs)
        variant_kwargs["icu_bed_delta"] = whatif_kwargs.get("icu_bed_delta", 0) + icu_bed_variant

        variant_result = whatif_fn(**variant_kwargs)

        baseline_unsafe = float(opt_result.details.get("total_unsafe_excess", 0.0))
        variant_unsafe = float(variant_result.get("total_unsafe_excess", 0.0))

        pct_change = ((variant_unsafe - baseline_unsafe) / baseline_unsafe * 100.0) if baseline_unsafe > 0 else 0.0

        summary = (
            f"Tested +{icu_bed_variant} ICU beds: unsafe excess would change from "
            f"{baseline_unsafe:.2f} to {variant_unsafe:.2f} ({pct_change:+.1f}%)."
        )
        return AgentStepResult(
            self.name,
            summary,
            {
                "icu_bed_variant": icu_bed_variant,
                "variant_unsafe_excess": variant_unsafe,
                "pct_change": pct_change,
            },
        )


class SupervisorAgent:
    """
    Final approval step: reviews the Explanation Agent's recommendation
    alongside the Scenario Agent's capacity-expansion test and issues a
    single approved directive -- the role a human ops lead plays when
    signing off on a recommendation before it's acted on.
    """

    name = "Supervisor Agent"

    def run(self, explanation_result: AgentStepResult, scenario_result: AgentStepResult) -> AgentStepResult:
        recommendation = explanation_result.details.get("recommendation", "No recommendation available.")
        pct_change = float(scenario_result.details.get("pct_change", 0.0))
        icu_bed_variant = scenario_result.details.get("icu_bed_variant", 0)

        directive = f"APPROVED: {recommendation}"
        if pct_change <= -10.0:
            directive += (
                f" Additionally, capacity expansion (+{icu_bed_variant} ICU beds) would reduce unsafe "
                f"excess by {abs(pct_change):.1f}% -- recommend evaluating for budget approval."
            )

        return AgentStepResult(self.name, directive, {"directive": directive})


def run_agent_pipeline(
    twin_horizon_final_day: Dict[str, Any],
    classify_fn: Callable[[float, float, float], tuple],
    whatif_fn: Callable[..., Dict[str, float]],
    whatif_kwargs: Dict[str, Any],
    icu_bed_variant: int = 15,
) -> List[AgentStepResult]:
    """
    Run the full Forecast -> Optimization -> Risk -> Explanation -> Scenario
    -> Supervisor agent chain and return the ordered trace of each agent's
    output. Each agent wraps existing platform logic (Digital Twin, live
    solver, stress classifier, diagnosis logic) -- this function is the
    orchestration layer, not a reimplementation of any of them.

    Note: this runs two live solves (Optimization Agent's baseline run, plus
    Scenario Agent's capacity-variant test), so it takes roughly twice as
    long as the 4-stage version.
    """
    trace: List[AgentStepResult] = []

    forecast_agent = ForecastAgent()
    forecast_result = forecast_agent.run(twin_horizon_final_day)
    trace.append(forecast_result)

    opt_agent = OptimizationAgent()
    opt_result = opt_agent.run(whatif_fn, whatif_kwargs)
    trace.append(opt_result)

    risk_agent = RiskAgent()
    risk_result = risk_agent.run(opt_result, classify_fn)
    trace.append(risk_result)

    explanation_agent = ExplanationAgent()
    explanation_result = explanation_agent.run(risk_result, opt_result)
    trace.append(explanation_result)

    scenario_agent = ScenarioAgent()
    scenario_result = scenario_agent.run(opt_result, whatif_fn, whatif_kwargs, icu_bed_variant=icu_bed_variant)
    trace.append(scenario_result)

    supervisor_agent = SupervisorAgent()
    supervisor_result = supervisor_agent.run(explanation_result, scenario_result)
    trace.append(supervisor_result)

    return trace