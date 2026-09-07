from __future__ import annotations

from src.validation.solver_readiness import check_gurobi_readiness, clear_solver_readiness_cache


def test_solver_readiness_returns_consistent_evidence() -> None:
    clear_solver_readiness_cache()
    readiness = check_gurobi_readiness()
    assert readiness.status
    assert readiness.check_seconds >= 0.0
    if readiness.license_verified:
        assert readiness.package_available is True
        assert readiness.version is not None
        assert readiness.status == "ready"
    if not readiness.package_available:
        assert readiness.license_verified is False
        assert readiness.status == "package_unavailable"
