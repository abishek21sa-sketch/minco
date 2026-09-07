from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation.forecast_baseline_validation import run_forecast_baseline_validation


def _write_validation_fixture(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for scenario, regime, offset in [
        ("normal", "normal", 0.0),
        ("respiratory_surge", "surge", 2.0),
    ]:
        for time_index in range(8):
            unsafe = offset + float(time_index)
            blocked = offset + float(time_index) / 2.0
            utilization = 0.9 + (0.05 * (time_index % 4)) + (offset / 20.0)
            rows.append(
                {
                    "time_index": time_index,
                    "scenario": scenario,
                    "dataset_source": "test_fixture",
                    "policy_name": "baseline_policy",
                    "design_name": "baseline_design",
                    "total_unsafe_excess": unsafe,
                    "total_unsafe_excess_rollmean_3": max(0.0, unsafe - 0.5),
                    "target_next_total_unsafe_excess": unsafe + 0.25,
                    "total_blocked_arrivals": blocked,
                    "total_blocked_arrivals_rollmean_3": max(0.0, blocked - 0.25),
                    "target_next_total_blocked_arrivals": blocked + 0.1,
                    "utilization_critical_flag": int(utilization > 1.2),
                    "target_next_utilization_critical_flag": int(utilization > 1.15),
                    "current_regime": regime,
                    "next_regime_label": "surge" if regime == "normal" else "crisis",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_forecast_baselines_use_temporal_and_scenario_holdouts(tmp_path: Path) -> None:
    data_path = tmp_path / "forecast_dataset_combined.csv"
    _write_validation_fixture(data_path)
    summary = run_forecast_baseline_validation(
        data_path=data_path,
        output_dir=tmp_path / "validation",
        manifest_dir=tmp_path / "manifests",
    )
    assert summary["dataset_rows"] > 0
    assert summary["result_rows"] > 0
    strategies = {item["strategy"] for item in summary["split_summaries"]}
    assert "chronological_holdout" in strategies
    assert "grouped_chronological_holdout" in strategies
    assert len(summary["scenarios_held_out"]) >= 2
    assert Path(summary["results_path"]).exists()
    assert Path(summary["manifest_path"]).exists()
