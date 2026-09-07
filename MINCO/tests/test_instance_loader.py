from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config.loader import load_healthcare_instance


def test_base_instance_loads(project_root: Path) -> None:
    instance = load_healthcare_instance(project_root / "data")
    assert instance.hospitals.df["hospital_id"].nunique() == 3
    assert len(instance.transitions.matrices) == 4
    assert len(instance.arrivals.df) == 84


def test_duplicate_transfer_lane_is_rejected(copied_data_dir: Path) -> None:
    path = copied_data_dir / "base_instance" / "transfer_lanes.csv"
    df = pd.read_csv(path)
    pd.concat([df, df.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate business keys"):
        load_healthcare_instance(copied_data_dir)


def test_unknown_hospital_reference_is_rejected(copied_data_dir: Path) -> None:
    path = copied_data_dir / "base_instance" / "arrivals.csv"
    df = pd.read_csv(path)
    df.loc[0, "hospital_id"] = "UNKNOWN"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown hospitals"):
        load_healthcare_instance(copied_data_dir)


@pytest.mark.parametrize("network_size", [10, 25, 50, 100, 200])
def test_all_synthetic_network_instances_load(project_root: Path, network_size: int) -> None:
    instance = load_healthcare_instance(project_root / "data" / "synthetic" / f"n{network_size}")
    assert instance.hospitals.df["hospital_id"].nunique() == network_size
