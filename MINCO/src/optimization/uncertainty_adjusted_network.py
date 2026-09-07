"""Truthfully named uncertainty-adjusted optimization compatibility layer."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from src.config.dataclasses import HealthcareInstance
from src.optimization.robust_uncertainty import RobustArrivalConfig


METHOD_ID = "arrival_stress_adjusted_nominal_lp"
METHOD_CLASSIFICATION = "uncertainty_adjusted_nominal_optimization"
FORMAL_ROBUST_OPTIMIZATION = False


def describe_uncertainty_adjusted_method(
    config: RobustArrivalConfig | None = None,
) -> dict[str, Any]:
    config = config or RobustArrivalConfig(default_rho=0.10)
    return {
        "method_id": METHOD_ID,
        "method_classification": METHOD_CLASSIFICATION,
        "formal_robust_optimization": FORMAL_ROBUST_OPTIMIZATION,
        "uncertainty_treatment": "arrival inflation before solving the nominal deterministic LP",
        "config": asdict(config),
        "approved_name": "uncertainty-adjusted network optimization",
        "legacy_name": "robust_optimized_network",
    }


def build_uncertainty_adjusted_network_policy_snapshot(
    instance: HealthcareInstance,
    config: RobustArrivalConfig | None = None,
) -> dict[str, Any]:
    """Build the existing arrival-stress policy under an accurate public name.

    The legacy robust builder remains available for backward compatibility, but
    this wrapper explicitly declares that the formulation is not a formal robust
    counterpart with an uncertainty set or adversarial subproblem.
    """
    from src.optimization.robust_minco import build_robust_optimized_network_policy_snapshot

    config = config or RobustArrivalConfig(default_rho=0.10)
    snapshot = build_robust_optimized_network_policy_snapshot(instance, config)
    metadata = describe_uncertainty_adjusted_method(config)
    snapshot["legacy_policy_name"] = "robust_optimized_network"
    snapshot["policy_name"] = "uncertainty_adjusted_network"
    snapshot["method_metadata"] = metadata
    metrics = snapshot.get("model_metrics")
    if isinstance(metrics, pd.DataFrame):
        metrics = metrics.copy()
        if metrics.empty:
            metrics = pd.DataFrame([{}])
        metrics["policy_name"] = "uncertainty_adjusted_network"
        metrics["method_id"] = METHOD_ID
        metrics["method_classification"] = METHOD_CLASSIFICATION
        metrics["formal_robust_optimization"] = False
        snapshot["model_metrics"] = metrics
    return snapshot
