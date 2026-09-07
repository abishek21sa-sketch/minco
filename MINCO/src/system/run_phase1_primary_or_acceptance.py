"""Windows/laptop acceptance gate for the primary Julia/JuMP/Gurobi runtime."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from src.config.paths import PROJECT_ROOT, RESULTS_DIR
from src.decision_math.stochastic_milp_oracle import make_tiny_oracle_instance, solve_small_stochastic_milp
from src.optimization_julia.bridge import solve_with_julia_gurobi

REPORT_PATH = RESULTS_DIR / "validation" / "phase1_mathematical_engine" / "primary_julia_or_acceptance.json"


def run_acceptance() -> dict:
    julia = shutil.which("julia")
    if not julia:
        raise RuntimeError("Julia is not installed or not on PATH. Run Julia installation/setup before this gate.")
    instance = make_tiny_oracle_instance()
    oracle = solve_small_stochastic_milp(instance)
    primary = solve_with_julia_gurobi(instance, julia_executable=julia)
    primary_status = str(primary.payload.get("termination_status"))
    primary_objective = float(primary.payload.get("objective"))
    objective_difference = abs(primary_objective - oracle.objective)

    ph_script = PROJECT_ROOT / "julia" / "scripts" / "run_progressive_hedging.jl"
    ph_completed = subprocess.run(
        [julia, str(ph_script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    ph_passed = ph_completed.returncode == 0 and '"acceptance":"PASSED"' in ph_completed.stdout.replace(" ", "")

    checks = {
        "julia_gurobi_status_optimal": primary_status == "OPTIMAL",
        "julia_objective_matches_highs_oracle": objective_difference <= 1e-5,
        "progressive_hedging_acceptance_passed": ph_passed,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "primary_julia_gurobi": primary.payload,
        "independent_highs_objective": oracle.objective,
        "objective_absolute_difference": objective_difference,
        "progressive_hedging_stdout": ph_completed.stdout,
        "progressive_hedging_stderr": ph_completed.stderr,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = run_acceptance()
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
