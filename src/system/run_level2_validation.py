"""One-command local acceptance gate for MINCO Level 2 Work Package 1."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config.paths import RESULTS_DIR
from src.validation.forecast_baseline_validation import run_forecast_baseline_validation
from src.validation.solver_readiness import check_gurobi_readiness
from src.version import __version__


def main() -> None:
    validation_dir = RESULTS_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    solver = check_gurobi_readiness().to_dict()
    forecast = run_forecast_baseline_validation()
    passed = bool(solver["license_verified"] and forecast["result_rows"] > 0)

    payload = {
        "work_package": "level_2_wp1_validation_foundation",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "solver_readiness": solver,
        "forecast_baseline_validation": forecast,
        "required_follow_up": (
            []
            if passed
            else ["Verify the local Gurobi package/license and rerun this acceptance command."]
        ),
    }
    report_path = validation_dir / "level2_wp1_validation_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "report_path": str(report_path)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
