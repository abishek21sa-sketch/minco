from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import load_healthcare_instance


def test_network_optimizer_is_feasible_repeatable_and_contract_compliant(
    project_root: Path,
) -> None:
    pytest.importorskip("gurobipy")
    from src.baselines.policy_baselines import build_optimized_network_policy_snapshot
    from src.optimizer.model_builder import solve_network_capacity_model
    from src.validation.optimization_checks import (
        assert_objective_repeatable,
        assert_optimal_model,
        validate_network_solution_tables,
    )

    instance = load_healthcare_instance(project_root / "data")
    model_one, *_ = solve_network_capacity_model(instance, output_flag=0)
    model_two, *_ = solve_network_capacity_model(instance, output_flag=0)

    first = assert_optimal_model(model_one)
    second = assert_optimal_model(model_two)
    assert_objective_repeatable(first, second)

    snapshot = build_optimized_network_policy_snapshot(instance)
    checks = validate_network_solution_tables(snapshot, instance)
    assert checks["surge_bound_max_excess"] <= 1e-6
    assert checks["elective_balance_max_error"] <= 1e-6
    assert checks["transfer_capacity_max_excess"] <= 1e-6
