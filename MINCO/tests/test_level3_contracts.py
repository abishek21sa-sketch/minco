from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts.common import CONTRACT_VERSION, ExecutionContext
from src.contracts.metrics import OperationalMetrics
from src.contracts.scenario import ScenarioEvaluationRequest, ScenarioParameters


def test_contract_version_and_context_are_explicit() -> None:
    context = ExecutionContext(source="test")
    assert CONTRACT_VERSION == "1.0"
    assert context.correlation_id.startswith("corr_")
    assert context.source == "test"


def test_custom_scenario_requires_overrides() -> None:
    with pytest.raises(ValidationError):
        ScenarioEvaluationRequest(scenario_id="custom")


def test_operational_metrics_normalize_legacy_numeric_values() -> None:
    metrics = OperationalMetrics.from_mapping(
        {
            "total_unsafe_excess": 4,
            "max_utilization_ratio": 1.05,
            "n_replications": 3.0,
            "model_status": 2.0,
            "solve_runtime_seconds": 0.2,
        }
    )
    assert metrics.n_replications == 3
    assert metrics.model_status == 2
    assert metrics.total_unsafe_excess == 4.0


def test_scenario_parameter_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        ScenarioParameters(demand_surge_multiplier=6.0)
