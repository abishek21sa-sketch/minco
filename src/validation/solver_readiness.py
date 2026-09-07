"""Lightweight, evidence-producing Gurobi runtime and license preflight."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SolverReadiness:
    package_available: bool
    license_verified: bool
    status: str
    version: Optional[str]
    message: str
    check_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def check_gurobi_readiness() -> SolverReadiness:
    """Import Gurobi and solve a one-variable LP to verify the active license.

    The result is cached for the lifetime of the API process so readiness checks
    remain inexpensive after the first successful or failed preflight.
    """
    started = time.perf_counter()
    try:
        import gurobipy as gp
    except Exception as exc:
        return SolverReadiness(
            package_available=False,
            license_verified=False,
            status="package_unavailable",
            version=None,
            message=f"{type(exc).__name__}: {exc}",
            check_seconds=time.perf_counter() - started,
        )

    version = ".".join(str(part) for part in gp.gurobi.version())
    env = None
    model = None
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("minco_readiness_preflight", env=env)
        model.Params.OutputFlag = 0
        x = model.addVar(lb=0.0, name="x")
        model.setObjective(x, gp.GRB.MINIMIZE)
        model.optimize()
        verified = model.Status == gp.GRB.OPTIMAL and model.SolCount >= 1
        return SolverReadiness(
            package_available=True,
            license_verified=verified,
            status="ready" if verified else "solver_not_optimal",
            version=version,
            message=(
                "Gurobi package and license verified by a one-variable optimization solve."
                if verified
                else f"Preflight solve ended with status={model.Status}, solutions={model.SolCount}."
            ),
            check_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return SolverReadiness(
            package_available=True,
            license_verified=False,
            status="license_or_runtime_unavailable",
            version=version,
            message=f"{type(exc).__name__}: {exc}",
            check_seconds=time.perf_counter() - started,
        )
    finally:
        if model is not None:
            model.dispose()
        if env is not None:
            env.dispose()


def clear_solver_readiness_cache() -> None:
    """Clear the process-local readiness cache, primarily for tests."""
    check_gurobi_readiness.cache_clear()
