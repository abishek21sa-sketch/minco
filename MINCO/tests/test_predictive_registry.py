from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.predictive_model_governance import _train_and_register


def test_rejected_task_registry_does_not_create_model_artifact(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "time_index": [0, 1, 2],
            "current_regime": ["normal", "surge", "crisis"],
            "current_regime_code": [0, 1, 2],
            "next_regime_label": ["surge", "crisis", "crisis"],
        }
    )
    data_path = tmp_path / "dataset.csv"
    df.to_csv(data_path, index=False)
    decisions = [
        {
            "task_name": "next_regime",
            "status": "rejected",
            "selected_model": None,
            "reason": "test",
        }
    ]
    registry, artifacts = _train_and_register(
        df,
        {"next_regime": ["time_index", "current_regime", "current_regime_code"]},
        decisions,
        data_path=data_path,
        model_dir=tmp_path / "models",
    )
    assert artifacts == []
    assert registry.loc[0, "governance_status"] == "rejected"
    assert registry.loc[0, "deployment_scope"] == "not_deployable"
    assert pd.isna(registry.loc[0, "model_path"])
