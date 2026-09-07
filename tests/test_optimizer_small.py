from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import load_healthcare_instance


def test_network_optimizer_has_feasible_solution_when_gurobi_is_available(
    project_root: Path,
) -> None:
    gp = pytest.importorskip("gurobipy")
    try:
        model = gp.Model("license_preflight")
        x = model.addVar(lb=0.0)
        model.setObjective(x)
        model.optimize()
    except Exception as exc:
        pytest.skip(f"Gurobi package is present but no usable license/runtime is available: {exc}")

    from src.optimizer.model_builder import solve_network_capacity_model

    instance = load_healthcare_instance(project_root / "data")
    solved, *_ = solve_network_capacity_model(instance, output_flag=0)
    assert solved.SolCount >= 1
    assert solved.Status == gp.GRB.OPTIMAL
