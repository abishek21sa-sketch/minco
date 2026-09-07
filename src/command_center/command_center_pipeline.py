from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.analysis.stress_classification import classify_stress_label
from src.command_center.alert_engine import evaluate_alerts, highest_alert_level
from src.command_center.executive_brief_generator import generate_executive_brief
from src.command_center.state_monitor import get_current_state
from src.storage.audit_repository import log_alerts, log_decision_run, log_recommendation


RESULTS_DIR = Path("results/command_center")


def run_command_center_pipeline(
    data_dir: Path = Path("data"),
    icu_bed_delta: int = 0,
    transfer_capacity_multiplier: float = 1.0,
    demand_surge_multiplier: float = 1.0,
    n_replications: int = 50,
    scenario_label: str = "current network state",
) -> dict:
    """
    monitor -> detect -> recommend -> audit, end to end.

    Runs a live solve for the current network state, evaluates it against the
    platform's shared alert thresholds, generates an executive brief, and
    writes all three as audit-trail artifacts:
      results/command_center/current_status.json
      results/command_center/alerts.csv
      results/command_center/executive_brief.md
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Command Center: reading current state (live solve) ---")
    state = get_current_state(
        data_dir=data_dir,
        icu_bed_delta=icu_bed_delta,
        transfer_capacity_multiplier=transfer_capacity_multiplier,
        demand_surge_multiplier=demand_surge_multiplier,
        n_replications=n_replications,
    )

    print("--- Command Center: evaluating alerts ---")
    alerts = evaluate_alerts(state, classify_fn=classify_stress_label)
    overall_level = highest_alert_level(alerts)

    print("--- Command Center: generating executive brief ---")
    brief_text = generate_executive_brief(state, alerts, scenario_label=scenario_label)

    timestamp = datetime.now(timezone.utc).isoformat()

    status_payload = {
        "timestamp": timestamp,
        "scenario_label": scenario_label,
        "overall_alert_level": overall_level,
        "state": state,
        "inputs": {
            "icu_bed_delta": icu_bed_delta,
            "transfer_capacity_multiplier": transfer_capacity_multiplier,
            "demand_surge_multiplier": demand_surge_multiplier,
            "n_replications": n_replications,
        },
    }

    status_path = RESULTS_DIR / "current_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status_payload, f, indent=2)

    alerts_path = RESULTS_DIR / "alerts.csv"
    file_exists = alerts_path.exists()
    with open(alerts_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "level", "trigger", "message"])
        if not file_exists:
            writer.writeheader()
        for alert in alerts:
            writer.writerow(
                {
                    "timestamp": alert.timestamp,
                    "level": alert.level,
                    "trigger": alert.trigger,
                    "message": alert.message,
                }
            )

    brief_path = RESULTS_DIR / "executive_brief.md"
    with open(brief_path, "w", encoding="utf-8") as f:
        f.write(brief_text)

    print(f"\nOverall alert level: {overall_level}")
    print(f"Saved: {status_path}")
    print(f"Appended: {alerts_path}")
    print(f"Saved: {brief_path}")

    print("--- Command Center: writing to audit database ---")
    run_id = log_decision_run(
        scenario=scenario_label,
        overall_alert=overall_level,
        pipeline_mode="command_center",
        runtime_seconds=state.get("solve_runtime_seconds"),
    )
    log_recommendation(
        run_id=run_id,
        selected_policy="optimized_network",
        predicted_unsafe_excess=state.get("total_unsafe_excess"),
    )
    log_alerts(run_id=run_id, alerts=alerts)
    print(f"Audit record saved: {run_id}")

    return {
        "run_id": run_id,
        "state": state,
        "alerts": [asdict(a) for a in alerts],
        "overall_alert_level": overall_level,
        "brief_text": brief_text,
    }


if __name__ == "__main__":
    run_command_center_pipeline()