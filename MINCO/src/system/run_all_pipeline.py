from __future__ import annotations

from pathlib import Path
from datetime import datetime
import subprocess
import sys
import json
import time

import pandas as pd


RESULTS_DIR = Path("results")
SYSTEM_DIR = RESULTS_DIR / "system_pipeline"
SYSTEM_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_LOG_PATH = SYSTEM_DIR / "run_all_pipeline_log.csv"
PIPELINE_SUMMARY_PATH = SYSTEM_DIR / "run_all_pipeline_summary.json"


PIPELINE_STEPS = [
    {
        "step_id": "S1",
        "name": "forecast_models_leakage_safe",
        "module": "src.ai.forecast_models_leakage_safe",
        "critical": True,
        "description": "Train leakage-safe forecasting models",
    },
    {
        "step_id": "S2",
        "name": "unsafe_excess_risk_classifier",
        "module": "src.ai.unsafe_excess_risk_classifier",
        "critical": True,
        "description": "Train unsafe-risk classification models",
    },
    {
        "step_id": "S3",
        "name": "regime_distribution_audit",
        "module": "src.analysis.regime_distribution_audit",
        "critical": True,
        "description": "Audit original regime imbalance",
    },
    {
        "step_id": "S4",
        "name": "regime_recalibration_plan",
        "module": "src.analysis.regime_recalibration_plan",
        "critical": True,
        "description": "Build calibrated regime labels",
    },
    {
        "step_id": "S5",
        "name": "calibrated_regime_forecast",
        "module": "src.ai.calibrated_regime_forecast",
        "critical": True,
        "description": "Current-state calibrated regime assignment",
    },
    {
        "step_id": "S6",
        "name": "calibrated_regime_forecast_lagged",
        "module": "src.ai.calibrated_regime_forecast_lagged",
        "critical": True,
        "description": "Strict lagged calibrated regime forecasting",
    },
    {
        "step_id": "S7",
        "name": "run_full_pipeline",
        "module": "src.system.run_full_pipeline",
        "critical": True,
        "description": "Execute integrated AI-OR decision pipeline",
    },
    {
        "step_id": "S8",
        "name": "ai_layer_summary",
        "module": "src.analysis.ai_layer_summary",
        "critical": True,
        "description": "Build AI layer evidence summary",
    },
    {
        "step_id": "S9",
        "name": "manuscript_package_v2",
        "module": "src.analysis.manuscript_package_v2",
        "critical": True,
        "description": "Build manuscript-ready package",
    },
]


def run_step(step: dict) -> dict:
    start = time.time()

    command = [
        sys.executable,
        "-m",
        step["module"],
    ]

    print("\n" + "=" * 80)
    print(f"RUNNING {step['step_id']} :: {step['name']}")
    print("=" * 80)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        runtime = time.time() - start

        success = result.returncode == 0

        status = "success" if success else "failed"

        print(result.stdout)

        if result.stderr.strip():
            print("\n--- STDERR ---")
            print(result.stderr)

        return {
            "step_id": step["step_id"],
            "name": step["name"],
            "module": step["module"],
            "description": step["description"],
            "critical": step["critical"],
            "status": status,
            "return_code": result.returncode,
            "runtime_seconds": round(runtime, 3),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }

    except Exception as e:
        runtime = time.time() - start

        return {
            "step_id": step["step_id"],
            "name": step["name"],
            "module": step["module"],
            "description": step["description"],
            "critical": step["critical"],
            "status": "exception",
            "return_code": -999,
            "runtime_seconds": round(runtime, 3),
            "stdout_tail": "",
            "stderr_tail": str(e),
        }


def build_summary(log_df: pd.DataFrame) -> dict:
    n_total = len(log_df)

    n_success = int((log_df["status"] == "success").sum())

    n_failed = int(
        (~log_df["status"].isin(["success"])).sum()
    )

    critical_failures = log_df[
        (log_df["critical"] == True)
        & (log_df["status"] != "success")
    ]

    failed_steps = (
        log_df[log_df["status"] != "success"]["name"]
        .astype(str)
        .tolist()
    )

    total_runtime = float(log_df["runtime_seconds"].sum())

    overall_status = (
        "success"
        if len(critical_failures) == 0
        else "critical_failure"
    )

    return {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall_status,
        "n_total_steps": n_total,
        "n_successful_steps": n_success,
        "n_failed_steps": n_failed,
        "failed_steps": failed_steps,
        "critical_failures": critical_failures["name"].tolist(),
        "total_runtime_seconds": round(total_runtime, 3),
        "output_dir": str(SYSTEM_DIR),
    }


def print_final_summary(summary: dict, log_df: pd.DataFrame) -> None:
    print("\n")
    print("=" * 80)
    print("FULL PIPELINE SUMMARY")
    print("=" * 80)

    print("\n=== Overall ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Step Results ===")

    display_cols = [
        "step_id",
        "name",
        "status",
        "runtime_seconds",
        "return_code",
    ]

    print(log_df[display_cols].to_string(index=False))

    successful = log_df[log_df["status"] == "success"]

    failed = log_df[log_df["status"] != "success"]

    print("\n=== Success Count ===")
    print(f"{len(successful)} / {len(log_df)}")

    if not failed.empty:
        print("\n=== Failed Steps ===")
        print(
            failed[
                [
                    "step_id",
                    "name",
                    "status",
                    "stderr_tail",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    print("\nFULL AI-OR REPRODUCIBILITY PIPELINE")
    print("=" * 80)

    rows = []

    pipeline_start = time.time()

    for step in PIPELINE_STEPS:
        result = run_step(step)

        rows.append(result)

        if (
            result["status"] != "success"
            and step["critical"]
        ):
            print("\nCRITICAL STEP FAILED.")
            print("Stopping pipeline execution.")
            break

    total_runtime = time.time() - pipeline_start

    log_df = pd.DataFrame(rows)

    summary = build_summary(log_df)

    summary["wall_clock_runtime_seconds"] = round(
        total_runtime,
        3,
    )

    log_df.to_csv(PIPELINE_LOG_PATH, index=False)

    PIPELINE_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print_final_summary(summary, log_df)

    print("\nSaved:")
    print(PIPELINE_LOG_PATH)
    print(PIPELINE_SUMMARY_PATH)


if __name__ == "__main__":
    main()