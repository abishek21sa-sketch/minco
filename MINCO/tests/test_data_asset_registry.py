from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.data_asset_registry import build_data_asset_registry


def test_data_asset_registry_hashes_and_classifies_assets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    base = data_dir / "base_instance"
    base.mkdir(parents=True)
    source = base / "arrivals.csv"
    pd.DataFrame({"day": [1], "hospital_id": ["H1"], "cohort": ["C1"], "arrivals": [3]}).to_csv(
        source, index=False
    )

    summary = build_data_asset_registry(
        asset_paths=[source], data_dir=data_dir, results_dir=results_dir
    )
    registry = pd.read_csv(results_dir / "data_registry" / "data_asset_registry.csv")

    assert summary["status"] == "passed"
    assert summary["asset_count"] == 1
    assert registry.loc[0, "source_classification"] == "synthetic_reference_case"
    assert len(registry.loc[0, "sha256"]) == 64
