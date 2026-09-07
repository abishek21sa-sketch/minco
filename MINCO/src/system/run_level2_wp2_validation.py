"""One-command acceptance gate for MINCO Level 2 Work Package 2."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config.paths import RESULTS_DIR
from src.validation.claim_governance import run_claim_governance_validation
from src.validation.optimization_sensitivity import run_optimization_sensitivity_validation
from src.validation.simulation_validation import run_simulation_validation
from src.validation.solver_readiness import check_gurobi_readiness
from src.version import __version__


def main() -> None:
    validation_dir = RESULTS_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    solver = check_gurobi_readiness().to_dict()
    simulation = run_simulation_validation()
    claims = run_claim_governance_validation()
    optimization = (
        run_optimization_sensitivity_validation()
        if solver["license_verified"]
        else {"status": "blocked", "reason": solver["message"]}
    )
    passed = bool(
        simulation["status"] == "passed"
        and claims["status"] == "passed"
        and optimization["status"] == "passed"
    )
    payload = {
        "work_package": "level_2_wp2_simulation_uncertainty_robustness",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "solver_readiness": solver,
        "simulation_validation": simulation,
        "claim_governance": claims,
        "optimization_sensitivity": optimization,
        "method_naming": {
            "approved": "uncertainty-adjusted network optimization",
            "legacy": "robust_optimized_network",
            "formal_robust_optimization": False,
        },
    }
    report_path = validation_dir / "level2_wp2_validation_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "report_path": str(report_path)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
