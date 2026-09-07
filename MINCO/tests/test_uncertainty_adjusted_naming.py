from __future__ import annotations

from src.optimization.uncertainty_adjusted_network import describe_uncertainty_adjusted_method


def test_uncertainty_adjusted_method_is_not_misclassified_as_formal_robust() -> None:
    metadata = describe_uncertainty_adjusted_method()
    assert metadata["approved_name"] == "uncertainty-adjusted network optimization"
    assert metadata["formal_robust_optimization"] is False
    assert metadata["method_classification"] == "uncertainty_adjusted_nominal_optimization"
