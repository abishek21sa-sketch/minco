"""
run_demo_pipeline.py

Fast inference-mode pipeline. Loads all pre-saved model artifacts and
results from the full training run instead of retraining anything, then
re-runs only the live decision/optimization step, AI summary, and report
packaging.

Target runtime: < 30 seconds (the three steps it actually runs combined
took ~12 seconds in the verified full-pipeline run).

Use this as the normal daily/demo command:
    python -m src.system.run_demo_pipeline

Use run_all_pipeline.py only when you need to retrain models from scratch:
    python -m src.system.run_all_pipeline
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


REQUIRED_ARTIFACTS = [
    "results/ai_models/calibrated_regime_forecast__gradient_boosting.pkl",
    "results/ai_reports/calibrated_regime_forecast_summary.json",
    "results/ai_reports/unsafe_excess_risk_classifier_summary.json",
    "results/tables/regime_recalibration_row_level.csv",
]

DEMO_STEPS = [
    {
        "step_id": "D1",
        "name": "Verify saved model artifacts",
        "module": None,  # handled inline, no subprocess
    },
    {
        "step_id": "D2",
        "name": "Run live optimization + decision pipeline",
        "module": "src.system.run_full_pipeline",
    },
    {
        "step_id": "D3",
        "name": "Generate AI-layer summary",
        "module": "src.analysis.ai_layer_summary",
    },
    {
        "step_id": "D4",
        "name": "Package results report",
        "module": "src.analysis.manuscript_package_v2",
    },
]


def verify_artifacts() -> dict:
    missing = []
    for path_str in REQUIRED_ARTIFACTS:
        if not (ROOT_DIR / path_str).exists():
            missing.append(path_str)

    if missing:
        return {
            "success": False,
            "message": (
                f"{len(missing)} required artifact(s) missing -- "
                f"run `python -m src.system.run_all_pipeline` first to generate them.\n"
                + "\n".join(f"  MISSING: {p}" for p in missing)
            ),
        }
    return {
        "success": True,
        "message": f"All {len(REQUIRED_ARTIFACTS)} required artifacts verified.",
    }


def run_module_step(module: str) -> dict:
    import subprocess

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )
    elapsed = time.perf_counter() - start
    success = result.returncode == 0
    if not success:
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
    return {"success": success, "runtime_seconds": elapsed}


def run_demo_pipeline() -> None:
    pipeline_start = time.perf_counter()
    log = []

    print("=" * 70)
    print("MERIDIAN HEALTH NETWORK — DEMO PIPELINE")
    print("(inference mode: loads saved models, skips ML retraining)")
    print("=" * 70)

    for step in DEMO_STEPS:
        step_id = step["step_id"]
        name = step["name"]
        print(f"\n{'=' * 70}")
        print(f"RUNNING {step_id}: {name}")
        print("=" * 70)

        if step["module"] is None:
            result = verify_artifacts()
            elapsed = 0.0
        else:
            result = run_module_step(step["module"])
            elapsed = result.get("runtime_seconds", 0.0)

        success = result["success"]
        status = "SUCCESS" if success else "FAILED"
        print(f"{status} ({elapsed:.1f}s): {name}")

        if not success:
            print(f"\nERROR: {result.get('message', 'step failed')}")
            print("\nDemo pipeline aborted. Check error above.")
            sys.exit(1)

        log.append({
            "step_id": step_id,
            "name": name,
            "module": step["module"],
            "runtime_seconds": round(elapsed, 3),
            "status": status,
        })

    total = time.perf_counter() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"DEMO PIPELINE COMPLETE — {total:.1f}s total ({len(log)} steps)")
    print("=" * 70)

    summary = {
        "pipeline": "demo (inference mode)",
        "total_runtime_seconds": round(total, 3),
        "n_steps": len(log),
        "steps": log,
    }
    out_path = ROOT_DIR / "results/system_pipeline/run_demo_pipeline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {out_path}")

    print("\nKey outputs refreshed:")
    print("  results/system_pipeline/full_pipeline_decision_report.json")
    print("  results/command_center/ (run make command-center to update)")
    print("\nLaunch dashboard:")
    print("  python -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    run_demo_pipeline()