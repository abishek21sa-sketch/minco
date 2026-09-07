from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dashboard.live_sandbox import clone_instance_with_whatif
from src.config.loader import load_healthcare_instance
from src.optimizer.transfer_utils import build_transfer_capacity_lookup
from src.scenarios.network_generator import generate_synthetic_network


def test_transfer_capacity_is_required_and_nonnegative(copied_data_dir: Path) -> None:
    path = copied_data_dir / "base_instance" / "transfer_lanes.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=["transfer_capacity"])
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_healthcare_instance(copied_data_dir)

    # Restore a malformed capacity field and prove numeric validation catches it.
    original = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "base_instance" / "transfer_lanes.csv"
    )
    original.loc[0, "transfer_capacity"] = -1.0
    original.to_csv(path, index=False)
    with pytest.raises(ValueError, match="negative values"):
        load_healthcare_instance(copied_data_dir)


def test_whatif_scales_capacity_without_changing_topology(project_root: Path) -> None:
    instance = load_healthcare_instance(project_root / "data")
    original = instance.transfer_lanes.df.sort_values(["from_hospital", "to_hospital"]).reset_index(
        drop=True
    )

    adjusted = clone_instance_with_whatif(instance, transfer_capacity_multiplier=0.8)
    lanes = adjusted.transfer_lanes.df.sort_values(["from_hospital", "to_hospital"]).reset_index(
        drop=True
    )

    assert lanes["allowed"].tolist() == original["allowed"].tolist()
    assert lanes["transfer_capacity"].tolist() == pytest.approx(
        (original["transfer_capacity"].astype(float) * 0.8).tolist()
    )
    assert (lanes["transfer_capacity"] > 0).all()


def test_transfer_capacity_lookup_and_generator_are_explicit(project_root: Path) -> None:
    instance = load_healthcare_instance(project_root / "data")
    lookup = build_transfer_capacity_lookup(instance)
    assert lookup[("H1", "H2")] == pytest.approx(8.0)
    assert lookup[("H2", "H1")] == pytest.approx(4.0)
    assert lookup[("H3", "H2")] == pytest.approx(2.0)

    generated = generate_synthetic_network(10)
    lanes = generated["transfer_lanes"]
    assert "transfer_capacity" in lanes.columns
    assert (lanes["transfer_capacity"] > 0).all()
    assert set(lanes["allowed"].unique()) == {1}
