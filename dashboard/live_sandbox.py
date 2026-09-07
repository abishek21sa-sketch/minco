from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Optional

from src.config.dataclasses import (
    ArrivalsData,
    CapacitiesData,
    HealthcareInstance,
    TransferLanesData,
)
from src.config.loader import load_healthcare_instance
from src.optimization.action_summary import summarize_policy_actions


def load_base_instance(data_dir: Path) -> HealthcareInstance:
    """Load the Meridian base instance from the given data directory."""
    return load_healthcare_instance(data_dir=data_dir)


def clone_instance_with_whatif(
    instance: HealthcareInstance,
    icu_bed_delta: int = 0,
    transfer_capacity_multiplier: float = 1.0,
    demand_surge_multiplier: float = 1.0,
) -> HealthcareInstance:
    """
    Return a copy of the instance with network-wide what-if adjustments applied.

    icu_bed_delta: total ICU beds added across the network, split proportionally
        across hospitals by their current ICU capacity share.
    transfer_capacity_multiplier: scales the allowed transfer-lane capacity.
    demand_surge_multiplier: scales patient arrivals.
    """
    capacities_df = instance.capacities.df.copy()
    hospital_col = "hospital_id" if "hospital_id" in capacities_df.columns else "hospital"
    resource_col = "resource" if "resource" in capacities_df.columns else "unit"
    capacity_col = "base_capacity" if "base_capacity" in capacities_df.columns else "capacity"

    capacities_df[capacity_col] = capacities_df[capacity_col].astype(float)

    if icu_bed_delta != 0:
        icu_mask = capacities_df[resource_col].astype(str).str.upper() == "ICU"
        total_icu = capacities_df.loc[icu_mask, capacity_col].sum()
        if total_icu > 0:
            share = capacities_df.loc[icu_mask, capacity_col] / total_icu
            capacities_df.loc[icu_mask, capacity_col] = (
                capacities_df.loc[icu_mask, capacity_col] + share * icu_bed_delta
            )

    transfer_df = instance.transfer_lanes.df.copy()
    if "transfer_capacity" not in transfer_df.columns:
        raise ValueError(
            "transfer_lanes.transfer_capacity is required for transfer-capacity scenarios"
        )
    if transfer_capacity_multiplier != 1.0:
        transfer_df["transfer_capacity"] = (
            transfer_df["transfer_capacity"].astype(float) * transfer_capacity_multiplier
        )

    arrivals_df = instance.arrivals.df.copy()
    if demand_surge_multiplier != 1.0 and "arrivals" in arrivals_df.columns:
        arrivals_df["arrivals"] = arrivals_df["arrivals"].astype(float) * demand_surge_multiplier

    return HealthcareInstance(
        hospitals=instance.hospitals,
        capacities=CapacitiesData(df=capacities_df),
        surge_caps=instance.surge_caps,
        safe_thresholds=instance.safe_thresholds,
        transfer_lanes=TransferLanesData(df=transfer_df),
        costs=instance.costs,
        elective_bounds=instance.elective_bounds,
        arrivals=ArrivalsData(df=arrivals_df),
        transitions=instance.transitions,
    )


def run_live_whatif(
    instance: HealthcareInstance,
    icu_bed_delta: int = 0,
    transfer_capacity_multiplier: float = 1.0,
    demand_surge_multiplier: float = 1.0,
    n_replications: int = 50,
    base_seed: int = 123,
) -> Dict[str, float]:
    """
    Solve the network optimization model for a what-if instance and run
    stochastic replications, returning mean KPIs plus solve diagnostics.

    n_replications and base_seed default to the same values used to generate
    the precomputed scenario_suite_summary baseline (50 reps, seed 123), so
    that a what-if run with zero changes reproduces the baseline numbers
    instead of just landing on a different random sample.
    """
    start_time = time.perf_counter()

    whatif_instance = clone_instance_with_whatif(
        instance=instance,
        icu_bed_delta=icu_bed_delta,
        transfer_capacity_multiplier=transfer_capacity_multiplier,
        demand_surge_multiplier=demand_surge_multiplier,
    )

    # Keep non-solver data/scenario transformations importable without Gurobi.
    from src.baselines.policy_baselines import build_optimized_network_policy_snapshot
    from src.simulation.policy_runner import run_multiple_stochastic_replications

    policy_snapshot = build_optimized_network_policy_snapshot(whatif_instance)

    outputs = run_multiple_stochastic_replications(
        instance=whatif_instance,
        policy_snapshot=policy_snapshot,
        n_replications=n_replications,
        base_seed=base_seed,
    )

    kpi_df = outputs["kpi_replications"].copy()

    metric_cols = [
        "total_unsafe_excess",
        "total_overflow_excess",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_blocked_arrivals",
    ]
    metric_cols = [c for c in metric_cols if c in kpi_df.columns]

    summary: Dict[str, float] = {col: float(kpi_df[col].mean()) for col in metric_cols}
    summary["n_replications"] = float(n_replications)

    action_summary = summarize_policy_actions(policy_snapshot)
    summary.update(action_summary)
    summary["solve_runtime_seconds"] = time.perf_counter() - start_time

    return summary


_NEGATIVE_WORDS = r"(drop|reduce|decrease|cut|fail|disabl|fewer|less|lower|lose|down)"


def _sign_near(text: str, start: int, end: int) -> float:
    window = text[max(0, start - 20):end]
    return -1.0 if re.search(_NEGATIVE_WORDS, window) else 1.0


def parse_whatif_question(
    question: str,
    total_icu_capacity: float,
) -> Optional[Dict[str, float]]:
    """
    Small pattern-matched parser for free-text what-if questions, e.g.:
      "What if we added 20 ICU beds?"
      "What if ICU capacity increased by 15%?"
      "What happens if transfer capacity drops 30%?"
      "What if demand surged 25%?"

    Returns a dict of icu_bed_delta / transfer_capacity_multiplier /
    demand_surge_multiplier if the question looks like a what-if request,
    otherwise None. This is intentionally simple keyword/regex matching, not
    a language model -- it only needs to catch the patterns above to give an
    honest live answer instead of pretending to understand arbitrary text.
    """
    q = question.lower()

    icu_bed_delta = 0
    transfer_pct = 0.0
    demand_pct = 0.0
    matched = False

    bed_match = re.search(r"(\+?\d+)\s*(?:more\s*)?icu\s*beds?", q)
    if bed_match:
        icu_bed_delta = int(bed_match.group(1).replace("+", ""))
        matched = True
    else:
        icu_pct_match = re.search(r"icu[^%]*?(\d+)\s*%", q) or re.search(r"(\d+)\s*%[^%]*?icu", q)
        if icu_pct_match:
            pct = float(icu_pct_match.group(1))
            sign = _sign_near(q, icu_pct_match.start(), icu_pct_match.end())
            icu_bed_delta = round(total_icu_capacity * (sign * pct) / 100.0)
            matched = True

    transfer_match = re.search(r"transfer[^%]*?(\d+)\s*%", q)
    if transfer_match:
        pct = float(transfer_match.group(1))
        sign = _sign_near(q, transfer_match.start(), transfer_match.end())
        transfer_pct = sign * pct
        matched = True

    demand_match = re.search(r"(?:demand|surge|arrival)[^%]*?(\d+)\s*%", q)
    if demand_match:
        pct = float(demand_match.group(1))
        sign = _sign_near(q, demand_match.start(), demand_match.end())
        demand_pct = sign * pct
        matched = True

    if not matched:
        return None

    return {
        "icu_bed_delta": icu_bed_delta,
        "transfer_capacity_multiplier": 1.0 + transfer_pct / 100.0,
        "demand_surge_multiplier": 1.0 + demand_pct / 100.0,
    }