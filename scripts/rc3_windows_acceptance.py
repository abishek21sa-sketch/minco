from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    print("RUNNING:", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, env=env, check=True)


def main():
    py = sys.executable
    run(
        py,
        "-m",
        "pytest",
        "tests/phase1/test_flow_cvar_network_signature.py",
        "tests/phase1/test_flow_cvar_decision.py",
        "tests/test_flow_cvar_api.py",
        "tests/test_flow_cvar_ui_contract.py",
        "tests/test_fixed_first_stage_oracle.py",
        "-q",
    )
    run(py, "scripts/flow_cvar_signature_evidence.py")
    run(py, "scripts/flow_cvar_product_evidence.py")
    run(
        py,
        "-m",
        "py_compile",
        "src/minco4x/signature_algorithm.py",
        "src/api/main.py",
        "dashboard/app.py",
    )
    print("FLOW_CVAR_WINDOWS_ACCEPTANCE=PASS")


if __name__ == "__main__":
    main()
