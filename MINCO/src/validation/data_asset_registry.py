"""Versioned inventory of MINCO analytical data, evidence, and model assets."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from src.validation.run_manifest import new_run_id, sha256_file, write_run_manifest

REGISTRY_DIR = RESULTS_DIR / "data_registry"
REGISTRY_PATH = REGISTRY_DIR / "data_asset_registry.csv"
SUMMARY_PATH = RESULTS_DIR / "validation" / "data_asset_registry_summary.json"


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _classify(path: Path) -> tuple[str, str]:
    relative = _relative(path)
    parts = set(Path(path).parts)
    if "base_instance" in parts or "transitions" in parts:
        return "input_data", "synthetic_reference_case"
    if relative.startswith("results/ai_datasets/"):
        return "derived_dataset", "synthetic_generated"
    if relative.startswith("results/model_registry/models/"):
        return "model_artifact", "trained_on_synthetic_reference_case"
    if relative.startswith("results/model_registry/"):
        return "model_registry", "generated_governance_evidence"
    if relative.startswith("results/validation/"):
        return "validation_evidence", "generated_validation_evidence"
    if relative.startswith("results/run_manifests/"):
        return "run_manifest", "generated_reproducibility_evidence"
    return "other", "unclassified"


def _describe_file(path: Path) -> dict[str, object]:
    category, source_classification = _classify(path)
    rows = None
    columns = None
    schema = None
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        rows = int(len(frame))
        columns = int(len(frame.columns))
        schema = json.dumps({name: str(dtype) for name, dtype in frame.dtypes.items()}, sort_keys=True)
    return {
        "asset_id": sha256_file(path)[:16],
        "path": _relative(path),
        "category": category,
        "source_classification": source_classification,
        "bytes": int(path.stat().st_size),
        "rows": rows,
        "columns": columns,
        "schema": schema,
        "sha256": sha256_file(path),
        "status": "available",
    }


def default_asset_paths(
    *, data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR
) -> list[Path]:
    candidates: list[Path] = []
    for directory in [data_dir / "base_instance", data_dir / "transitions"]:
        if directory.exists():
            candidates.extend(sorted(directory.glob("*.csv")))
    forecast_dataset = results_dir / "ai_datasets" / "forecast_dataset_combined.csv"
    if forecast_dataset.exists():
        candidates.append(forecast_dataset)
    for path in [
        results_dir / "model_registry" / "predictive_model_registry.csv",
        results_dir / "validation" / "predictive_model_governance_summary.json",
        results_dir / "validation" / "level2_wp1_validation_report.json",
        results_dir / "validation" / "level2_wp2_validation_report.json",
        results_dir / "validation" / "level2_wp3_validation_report.json",
    ]:
        if path.exists():
            candidates.append(path)
    model_dir = results_dir / "model_registry" / "models"
    if model_dir.exists():
        candidates.extend(sorted(path for path in model_dir.glob("*") if path.is_file()))
    return sorted(set(Path(path) for path in candidates))


def build_data_asset_registry(
    *,
    asset_paths: Iterable[Path] | None = None,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    registry_path: Path | None = None,
    summary_path: Path | None = None,
    manifest_dir: Path | None = None,
) -> dict[str, object]:
    """Create a hash-addressed registry and an evidence summary."""
    registry_path = Path(registry_path or (results_dir / "data_registry" / REGISTRY_PATH.name))
    summary_path = Path(summary_path or (results_dir / "validation" / SUMMARY_PATH.name))
    manifest_dir = Path(manifest_dir or (results_dir / "run_manifests"))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    paths = list(asset_paths) if asset_paths is not None else default_asset_paths(
        data_dir=data_dir, results_dir=results_dir
    )
    missing = [_relative(Path(path)) for path in paths if not Path(path).exists()]
    rows = [_describe_file(Path(path)) for path in paths if Path(path).is_file()]
    registry = pd.DataFrame(rows)
    if registry.empty:
        raise RuntimeError("No MINCO assets were available for registration")
    registry = registry.sort_values(["category", "path"]).reset_index(drop=True)
    registry.to_csv(registry_path, index=False)

    synthetic_assets = int(
        registry["source_classification"].astype(str).str.contains("synthetic").sum()
    )
    real_hospital_assets = int(
        registry["source_classification"].astype(str).str.contains("real_hospital").sum()
    )
    run_id = new_run_id("data_registry")
    summary: dict[str, object] = {
        "run_id": run_id,
        "status": "passed" if not missing and real_hospital_assets == 0 else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_count": int(len(registry)),
        "category_counts": {
            str(key): int(value)
            for key, value in registry["category"].value_counts().to_dict().items()
        },
        "synthetic_or_generated_asset_count": synthetic_assets,
        "real_hospital_asset_count": real_hospital_assets,
        "missing_assets": missing,
        "claim_boundary": (
            "The bundled registry contains synthetic reference data and generated analytical "
            "evidence only. It does not contain real hospital operational data."
        ),
        "registry_path": _relative(registry_path),
        "registry_sha256": sha256_file(registry_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path, manifest_hash = write_run_manifest(
        run_id=run_id,
        run_type="data_asset_registry",
        parameters={"asset_count": len(registry)},
        input_paths=[Path(path) for path in paths if Path(path).exists()],
        output_paths=[registry_path],
        metrics={
            "asset_count": len(registry),
            "real_hospital_asset_count": real_hospital_assets,
            "missing_asset_count": len(missing),
        },
        notes=[summary["claim_boundary"]],
        manifest_dir=manifest_dir,
    )
    summary["manifest_path"] = str(manifest_path)
    summary["manifest_sha256"] = manifest_hash
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_data_asset_registry(), indent=2))
