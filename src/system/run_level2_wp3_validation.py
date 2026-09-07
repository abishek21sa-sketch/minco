"""One-command acceptance gate for MINCO Level 2 Work Package 3."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config.paths import RESULTS_DIR
from src.validation.predictive_model_governance import run_predictive_model_governance
from src.version import __version__


def main() -> None:
    validation_dir = RESULTS_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    governance = run_predictive_model_governance()
    decisions = governance["task_decisions"]
    required_tasks = {
        "next_unsafe_excess",
        "next_blocked_arrivals",
        "next_utilization_critical",
        "next_regime",
    }
    observed_tasks = {item["task_name"] for item in decisions}
    passed = (
        governance["status"] == "passed"
        and observed_tasks == required_tasks
        and bool(governance.get("manifest_sha256"))
    )

    payload = {
        "work_package": "level_2_wp3_predictive_model_validation_and_governance",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "predictive_model_governance": governance,
        "required_follow_up": (
            []
            if passed
            else ["Inspect the predictive governance outputs and resolve missing task evidence."]
        ),
    }
    report_path = validation_dir / "level2_wp3_validation_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "report_path": str(report_path)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
