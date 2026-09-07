"""Immutable run manifests with input/output hashes and runtime metadata."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.config.paths import PROJECT_ROOT, RESULTS_DIR
from src.version import __version__

RUN_MANIFEST_DIR = RESULTS_DIR / "run_manifests"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "validation") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def describe_path(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": _display_path(path), "exists": False}

    if path.is_file():
        return {
            "path": _display_path(path),
            "exists": True,
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    files = sorted(p for p in path.rglob("*") if p.is_file())
    aggregate = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        file_digest = sha256_file(file_path)
        total_bytes += file_path.stat().st_size
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "path": _display_path(path),
        "exists": True,
        "kind": "directory",
        "file_count": len(files),
        "bytes": total_bytes,
        "sha256": aggregate.hexdigest(),
    }


def _git_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _package_versions(names: Iterable[str]) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_run_manifest(
    *,
    run_id: str,
    run_type: str,
    parameters: Mapping[str, Any],
    input_paths: Iterable[Path] = (),
    output_paths: Iterable[Path] = (),
    metrics: Optional[Mapping[str, Any]] = None,
    model_info: Optional[Mapping[str, Any]] = None,
    notes: Optional[list[str]] = None,
    manifest_dir: Path = RUN_MANIFEST_DIR,
) -> tuple[Path, str]:
    """Write an atomic JSON run manifest and return its path and SHA-256."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{run_id}.json"

    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "run_type": run_type,
        "created_at": _now(),
        "minco_version": __version__,
        "git_commit": _git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "packages": _package_versions(
                ["numpy", "pandas", "fastapi", "pydantic", "scikit-learn", "gurobipy"]
            ),
        },
        "parameters": dict(parameters),
        "inputs": [describe_path(Path(path)) for path in input_paths],
        "outputs": [describe_path(Path(path)) for path in output_paths],
        "metrics": dict(metrics or {}),
        "model_info": dict(model_info or {}),
        "notes": list(notes or []),
    }

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path, sha256_file(manifest_path)
