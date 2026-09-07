"""Fast, solver-independent MINCO smoke pipeline.

This pipeline validates the data contract, deterministic forecast, capacity
comparison, and seeded stochastic simulation. It intentionally does not claim
to validate Gurobi-backed optimization.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config.loader import load_healthcare_instance
from src.config.paths import DATA_DIR, RESULTS_DIR
from src.markov.forecast_engine import run_forecast
from src.markov.forecast_validation import validate_forecast_against_capacity
from src.simulation.twin_engine import simulate_stochastic_twin
from src.version import __version__


def run_smoke(data_dir: Path = DATA_DIR, seed: int = 20260727) -> dict[str, Any]:
    instance = load_healthcare_instance(data_dir)
    forecast_a = run_forecast(instance)
    forecast_b = run_forecast(instance)
    forecast_repeatable = forecast_a["resource_demand"].equals(forecast_b["resource_demand"])
    validated = validate_forecast_against_capacity(forecast_a["resource_demand"], instance)

    sim_a = simulate_stochastic_twin(instance, rng=np.random.default_rng(seed))
    sim_b = simulate_stochastic_twin(instance, rng=np.random.default_rng(seed))
    simulation_repeatable = sim_a["state_trajectories"].equals(sim_b["state_trajectories"])

    checks = {
        "instance_loaded": True,
        "forecast_repeatable": forecast_repeatable,
        "simulation_repeatable": simulation_repeatable,
        "forecast_rows": int(len(forecast_a["resource_demand"])),
        "validated_capacity_rows": int(len(validated)),
        "simulation_rows": int(len(sim_a["state_trajectories"])),
        "max_expected_utilization": float(validated["utilization_ratio"].max()),
    }
    checks["status"] = (
        "passed"
        if all([checks["instance_loaded"], forecast_repeatable, simulation_repeatable])
        else "failed"
    )

    return {
        "pipeline": "minco-smoke",
        "version": __version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "seed": seed,
        "data_dir": str(Path(data_dir).resolve()),
        "checks": checks,
        "scope_note": "Solver-independent smoke validation; optimization requires a separate Gurobi gate.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "system_pipeline" / "smoke_report.json",
    )
    args = parser.parse_args()
    report = run_smoke(args.data_dir, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["checks"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
