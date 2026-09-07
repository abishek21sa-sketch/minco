from __future__ import annotations

import pytest


def test_optimization_sensitivity_monotonicity() -> None:
    pytest.importorskip("gurobipy")
    from src.validation.optimization_sensitivity import run_optimization_sensitivity_validation

    result = run_optimization_sensitivity_validation()
    assert result["status"] == "passed"
    assert all(result["checks"].values())
