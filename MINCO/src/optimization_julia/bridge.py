"""Python -> Julia/JuMP stochastic optimization bridge."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from src.config.paths import PROJECT_ROOT
from src.decision_math.stochastic_milp_oracle import StochasticCapacityInstance


def configured_solver_license_mode() -> str:
    """Return the declared solver posture for this deployment."""
    return (os.getenv("MINCO_SOLVER_LICENSE_MODE", "academic").strip().lower() or "academic")


def solver_license_notice() -> str:
    mode = configured_solver_license_mode()
    if mode == "academic":
        return "Academic research/development mode; commercial and operational use is not permitted by the license."
    return f"Declared solver license mode: {mode}. Verify the installed license terms before operational use."


@dataclass(frozen=True)
class JuliaSolveResult:
    payload: dict[str, Any]
    stdout: str
    stderr: str


@dataclass(frozen=True)
class JuliaReadiness:
    """Preflight result for the optional Julia/JuMP/Gurobi primary solver."""

    executable_available: bool
    project_manifest_available: bool
    dependencies_available: bool
    ready: bool
    status: str
    check_seconds: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable_available": self.executable_available,
            "project_manifest_available": self.project_manifest_available,
            "dependencies_available": self.dependencies_available,
            "ready": self.ready,
            "status": self.status,
            "check_seconds": self.check_seconds,
            "message": self.message,
            "license_mode": configured_solver_license_mode(),
            "license_notice": solver_license_notice(),
        }


def check_julia_readiness(
    *,
    julia_executable: str | None = None,
    timeout_seconds: float = 20.0,
) -> JuliaReadiness:
    """Verify the Julia project and required packages, not just the executable."""
    started = time.perf_counter()
    julia = julia_executable or shutil.which("julia")
    project_dir = PROJECT_ROOT / "julia"
    manifest_available = (project_dir / "Project.toml").exists()
    if not julia:
        return JuliaReadiness(
            executable_available=False,
            project_manifest_available=manifest_available,
            dependencies_available=False,
            ready=False,
            status="executable_unavailable",
            check_seconds=round(time.perf_counter() - started, 4),
            message="Julia executable not found; the SciPy/HiGHS verification oracle remains available.",
        )
    if not manifest_available:
        return JuliaReadiness(
            executable_available=True,
            project_manifest_available=False,
            dependencies_available=False,
            ready=False,
            status="project_manifest_unavailable",
            check_seconds=round(time.perf_counter() - started, 4),
            message=f"Julia project manifest missing: {project_dir / 'Project.toml'}",
        )

    try:
        completed = subprocess.run(
            [
                str(julia),
                "--startup-file=no",
                f"--project={project_dir}",
                "-e",
                "using JuMP; using Gurobi; using JSON3; println(\"MINCO_JULIA_READY\")",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return JuliaReadiness(
            executable_available=True,
            project_manifest_available=True,
            dependencies_available=False,
            ready=False,
            status="preflight_timeout",
            check_seconds=round(time.perf_counter() - started, 4),
            message=f"Julia package preflight exceeded {timeout_seconds:.1f}s. {solver_license_notice()}",
        )
    except OSError as exc:
        return JuliaReadiness(
            executable_available=True,
            project_manifest_available=True,
            dependencies_available=False,
            ready=False,
            status="preflight_error",
            check_seconds=round(time.perf_counter() - started, 4),
            message=f"Julia package preflight failed to start: {exc}",
        )

    ready = completed.returncode == 0 and "MINCO_JULIA_READY" in completed.stdout
    message = (
        f"Julia executable and JuMP/Gurobi/JSON3 project dependencies are ready for {configured_solver_license_mode()} research/development mode."
        if ready
        else "Julia is installed but the JuMP/Gurobi/JSON3 project environment is not ready. "
        "Run `julia julia/setup.jl` from the MINCO repository root, then retry. "
        f"{solver_license_notice()}"
    )
    return JuliaReadiness(
        executable_available=True,
        project_manifest_available=True,
        dependencies_available=ready,
        ready=ready,
        status="ready" if ready else "dependencies_unavailable",
        check_seconds=round(time.perf_counter() - started, 4),
        message=message,
    )


def instance_to_payload(instance: StochasticCapacityInstance) -> dict[str, Any]:
    instance.validate()
    return {
        "hospitals": list(instance.hospitals),
        "periods": list(instance.periods),
        "probabilities": instance.scenario_probabilities.tolist(),
        "demand": instance.demand.tolist(),
        "base_beds": instance.base_beds.tolist(),
        "safe_fraction": instance.safe_fraction.tolist(),
        "surge_beds": instance.surge_beds.tolist(),
        "surge_cost": instance.surge_cost.tolist(),
        "flex_beds_per_block": instance.flex_beds_per_block.tolist(),
        "flex_cost": instance.flex_cost.tolist(),
        "max_flex_blocks": instance.max_flex_blocks.tolist(),
        "deferral_limit": instance.elective_deferral_limit.tolist(),
        "transfer_capacity": instance.transfer_capacity.tolist(),
        "transfer_cost": instance.transfer_cost,
        "defer_cost": instance.defer_cost,
        "unsafe_cost": instance.unsafe_cost,
        "boarding_cost": instance.boarding_cost,
        "cvar_alpha": instance.cvar_alpha,
        "cvar_weight": instance.cvar_weight,
    }


def solve_with_julia_gurobi(
    instance: StochasticCapacityInstance,
    *,
    julia_executable: str | None = None,
    timeout_seconds: float = 180.0,
) -> JuliaSolveResult:
    julia = julia_executable or shutil.which("julia")
    if not julia:
        raise RuntimeError("Julia executable not found. Install Julia before using the primary optimizer.")
    readiness = check_julia_readiness(
        julia_executable=str(julia),
        timeout_seconds=min(timeout_seconds, 20.0),
    )
    if not readiness.ready:
        raise RuntimeError(f"Julia primary optimizer is not ready: {readiness.message}")
    script = PROJECT_ROOT / "julia" / "scripts" / "solve_json.jl"
    if not script.exists():
        raise RuntimeError(f"Julia bridge script missing: {script}")
    with tempfile.TemporaryDirectory(prefix="minco_julia_") as tmp:
        input_path = Path(tmp) / "instance.json"
        output_path = Path(tmp) / "solution.json"
        input_path.write_text(json.dumps(instance_to_payload(instance)), encoding="utf-8")
        completed = subprocess.run(
            [str(julia), str(script), str(input_path), str(output_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            remediation = ""
            if "Package " in completed.stderr and " not found in current path" in completed.stderr:
                remediation = " Re-run `julia julia/setup.jl` from the MINCO repository root, then retry acceptance."
            raise RuntimeError(
                "Julia/JuMP/Gurobi solve failed. "
                f"exit={completed.returncode}; stdout={completed.stdout}; stderr={completed.stderr}{remediation}"
            )
        if not output_path.exists():
            raise RuntimeError("Julia solver completed without producing a solution JSON file")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return JuliaSolveResult(payload=payload, stdout=completed.stdout, stderr=completed.stderr)
