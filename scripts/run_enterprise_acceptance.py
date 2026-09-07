"""Run the repeatable enterprise candidate-package acceptance checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_ROOT = ROOT.parent
REPORT_PATH = ROOT / "results" / "validation" / "enterprise_acceptance_report.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_step(name: str, args: list[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        return {
            "name": name,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "return_code": completed.returncode,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "status": "FAIL",
            "return_code": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "output_tail": f"Timed out after 900 seconds: {exc}",
        }
    except OSError as exc:
        return {
            "name": name,
            "status": "FAIL",
            "return_code": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "output_tail": f"Could not start command: {type(exc).__name__}: {exc}",
        }


def _registry_check() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from src.services.contract_registry import build_service_contract_registry

        result = build_service_contract_registry()
        ok = result.get("status") == "passed" and result.get("schema_count") == 20
        return {
            "name": "public_contract_registry",
            "status": "PASS" if ok else "FAIL",
            "schema_count": result.get("schema_count"),
            "schema_names": result.get("schema_names", []),
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:  # pragma: no cover - failure payload is the value here
        return {
            "name": "public_contract_registry",
            "status": "FAIL",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _api_check() -> dict[str, Any]:
    started = time.perf_counter()
    correlation_id = "enterprise-harness-20260905"
    try:
        from fastapi.testclient import TestClient

        from src.api.main import create_app

        client = TestClient(create_app())
        checks: dict[str, Any] = {}

        root = client.get("/")
        checks["/"] = {
            "status_code": root.status_code,
            "command_center_shell": "MINCO Command Center" in root.text,
        }

        openapi = client.get("/openapi.json")
        checks["/openapi.json"] = {
            "status_code": openapi.status_code,
            "decision_packet_route_present": "/v1/decision-packets/{run_id}"
            in openapi.json().get("paths", {}),
        }

        for path in (
            "/health",
            "/readiness",
            "/v1/audit/integrity",
            "/v1/operational-events/status",
        ):
            response = client.get(path, headers={"X-Correlation-ID": correlation_id})
            checks[path] = {
                "status_code": response.status_code,
                "correlation_id": response.headers.get("X-Correlation-ID"),
            }

        metrics = client.get(
            "/v1/observability/metrics",
            headers={"X-Correlation-ID": correlation_id},
        )
        prometheus = client.get("/metrics")
        evidence = client.get(
            "/v1/release-evidence",
            headers={"X-Correlation-ID": correlation_id},
        )
        readiness = client.get(
            "/v1/release-readiness",
            headers={"X-Correlation-ID": correlation_id},
        )
        checks["/v1/observability/metrics"] = {"status_code": metrics.status_code}
        checks["/metrics"] = {
            "status_code": prometheus.status_code,
            "prometheus_metric_present": "minco_api_requests_total" in prometheus.text,
        }
        evidence_payload = evidence.json()
        readiness_payload = readiness.json()
        checks["/v1/release-evidence"] = {
            "status_code": evidence.status_code,
            "attestation_status": evidence_payload.get("status"),
            "required_artifacts_hashed": evidence_payload.get("gates", {}).get(
                "required_artifacts_hashed"
            ),
            "missing_artifacts": evidence_payload.get("missing_artifacts", []),
        }
        checks["/v1/release-readiness"] = {
            "status_code": readiness.status_code,
            "status": readiness_payload.get("status"),
            "release_evidence_gate": readiness_payload.get("checks", {}).get(
                "release_evidence_attestation_verified"
            ),
            "autonomous_execution_permitted": readiness_payload.get(
                "autonomous_execution_permitted"
            ),
        }
        expected_statuses = all(
            item["status_code"] == 200 for item in checks.values() if "status_code" in item
        )
        ok = (
            expected_statuses
            and checks["/"]["command_center_shell"]
            and checks["/openapi.json"]["decision_packet_route_present"]
            and checks["/metrics"]["prometheus_metric_present"]
            and evidence_payload.get("status") == "VERIFIED"
            and evidence_payload.get("missing_artifacts") == []
            and readiness_payload.get("status") == "BLOCKED"
            and readiness_payload.get("checks", {}).get("release_evidence_attestation_verified")
            is True
            and readiness_payload.get("autonomous_execution_permitted") is False
        )
        return {
            "name": "api_governance_smoke",
            "status": "PASS" if ok else "FAIL",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "checks": checks,
        }
    except Exception as exc:  # pragma: no cover - failure payload is the value here
        return {
            "name": "api_governance_smoke",
            "status": "FAIL",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-product-runtime",
        action="store_true",
        help="Skip the Windows product-runtime acceptance command.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip the full pytest suite; not recommended for release review.",
    )
    args = parser.parse_args()
    steps: list[dict[str, Any]] = []
    python = sys.executable

    if not args.skip_product_runtime:
        steps.append(
            _run_step(
                "product_runtime_acceptance",
                ["cmd.exe", "/c", "RUN_PRODUCT_ACCEPTANCE.cmd"],
                PRODUCT_ROOT,
            )
        )
    if not args.skip_regression:
        steps.append(_run_step("python_regression", [python, "-m", "pytest", "-q"], ROOT))
    steps.append(
        _run_step(
            "phase2_backend_acceptance",
            [python, "-m", "src.system.run_phase2_backend_acceptance"],
            ROOT,
        )
    )
    steps.append(_registry_check())
    steps.append(
        _run_step(
            "persist_release_evidence",
            [python, "scripts/generate_release_evidence.py"],
            ROOT,
        )
    )
    steps.append(_api_check())

    status = "PASS" if all(step.get("status") == "PASS" for step in steps) else "FAIL"
    report = {
        "status": status,
        "generated_at": _now(),
        "suite": "MINCO enterprise candidate acceptance",
        "python": sys.version,
        "steps": steps,
        "claim_boundary": (
            "This verifies engineering and product-package controls for the synthetic Meridian "
            "reference case. It does not approve clinical, live-hospital, autonomous, or production use."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "report_path": str(REPORT_PATH)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
