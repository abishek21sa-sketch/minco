"""Forecast baseline validation under temporal and scenario-held-out splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from src.config.paths import RESULTS_DIR
from src.validation.run_manifest import RUN_MANIFEST_DIR, new_run_id, write_run_manifest
from src.validation.temporal_splits import (
    chronological_holdout,
    grouped_chronological_holdout,
    leave_one_group_out,
)

DATA_PATH = RESULTS_DIR / "ai_datasets" / "forecast_dataset_combined.csv"
OUTPUT_DIR = RESULTS_DIR / "validation"
RESULTS_PATH = OUTPUT_DIR / "forecast_baseline_validation.csv"
SUMMARY_PATH = OUTPUT_DIR / "forecast_baseline_validation_summary.json"


def _regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_array = actual.to_numpy(dtype=float)
    predicted_array = predicted.to_numpy(dtype=float)
    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "r2": float(r2_score(actual_array, predicted_array)),
    }


def _classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_array = actual.astype(str).to_numpy()
    predicted_array = predicted.astype(str).to_numpy()
    return {
        "accuracy": float(accuracy_score(actual_array, predicted_array)),
        "f1_macro": float(f1_score(actual_array, predicted_array, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(actual_array, predicted_array, average="weighted", zero_division=0)
        ),
    }


def _majority_predictor(train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.Series:
    majority = train[target].astype(str).mode(dropna=True)
    if majority.empty:
        raise ValueError(f"Cannot calculate majority baseline for {target}")
    return pd.Series(majority.iloc[0], index=test.index)


def _evaluate_split(
    df: pd.DataFrame,
    *,
    train_index: pd.Index,
    test_index: pd.Index,
    split_strategy: str,
    held_out_group: str | None = None,
) -> list[dict]:
    train = df.loc[train_index]
    test = df.loc[test_index]
    rows: list[dict] = []

    regression_baselines: dict[str, dict[str, str]] = {
        "target_next_total_unsafe_excess": {
            "persistence_current": "total_unsafe_excess",
            "rolling_mean_3": "total_unsafe_excess_rollmean_3",
        },
        "target_next_total_blocked_arrivals": {
            "persistence_current": "total_blocked_arrivals",
            "rolling_mean_3": "total_blocked_arrivals_rollmean_3",
        },
    }
    for target, baselines in regression_baselines.items():
        if target not in test.columns:
            continue
        for baseline_name, feature_col in baselines.items():
            if feature_col not in test.columns:
                continue
            valid = test[[target, feature_col]].dropna()
            if valid.empty:
                continue
            metrics = _regression_metrics(valid[target], valid[feature_col])
            rows.append(
                {
                    "split_strategy": split_strategy,
                    "held_out_group": held_out_group,
                    "task_type": "regression",
                    "target": target,
                    "baseline": baseline_name,
                    "n_train": len(train),
                    "n_test": len(valid),
                    **metrics,
                }
            )

    classification_specs: list[tuple[str, str, str]] = [
        (
            "target_next_utilization_critical_flag",
            "persistence_current",
            "utilization_critical_flag",
        ),
        ("next_regime_label", "regime_persistence", "current_regime"),
    ]
    for target, baseline_name, feature_col in classification_specs:
        if target not in test.columns or feature_col not in test.columns:
            continue
        valid = test[[target, feature_col]].dropna()
        if valid.empty:
            continue
        metrics = _classification_metrics(valid[target], valid[feature_col])
        rows.append(
            {
                "split_strategy": split_strategy,
                "held_out_group": held_out_group,
                "task_type": "classification",
                "target": target,
                "baseline": baseline_name,
                "n_train": len(train),
                "n_test": len(valid),
                **metrics,
            }
        )

        majority_test = test[[target]].dropna()
        if not majority_test.empty:
            majority_pred = _majority_predictor(train.dropna(subset=[target]), majority_test, target)
            majority_metrics = _classification_metrics(majority_test[target], majority_pred)
            rows.append(
                {
                    "split_strategy": split_strategy,
                    "held_out_group": held_out_group,
                    "task_type": "classification",
                    "target": target,
                    "baseline": "training_majority_class",
                    "n_train": len(train),
                    "n_test": len(majority_test),
                    **majority_metrics,
                }
            )
    return rows


def run_forecast_baseline_validation(
    *,
    data_path: Path = DATA_PATH,
    output_dir: Path = OUTPUT_DIR,
    manifest_dir: Path = RUN_MANIFEST_DIR,
) -> dict:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Forecast validation dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required = {"time_index", "scenario"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Forecast validation dataset is missing required columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    split_summaries: list[dict] = []

    train_idx, test_idx, metadata = chronological_holdout(df, test_fraction=0.25)
    result_rows.extend(
        _evaluate_split(
            df,
            train_index=train_idx,
            test_index=test_idx,
            split_strategy=metadata.strategy,
        )
    )
    split_summaries.append(metadata.to_dict())

    group_columns = [
        col
        for col in ["dataset_source", "scenario", "policy_name", "design_name"]
        if col in df.columns
    ]
    train_idx, test_idx, metadata = grouped_chronological_holdout(
        df,
        group_cols=group_columns,
        test_fraction=0.25,
    )
    result_rows.extend(
        _evaluate_split(
            df,
            train_index=train_idx,
            test_index=test_idx,
            split_strategy=metadata.strategy,
        )
    )
    split_summaries.append(metadata.to_dict())

    for scenario in sorted(df["scenario"].dropna().unique().tolist()):
        train_idx, test_idx = leave_one_group_out(
            df,
            group_col="scenario",
            held_out_value=scenario,
        )
        result_rows.extend(
            _evaluate_split(
                df,
                train_index=train_idx,
                test_index=test_idx,
                split_strategy="leave_one_scenario_out",
                held_out_group=str(scenario),
            )
        )

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError("No forecast baseline validation rows were produced")
    results_path = output_dir / RESULTS_PATH.name
    summary_path = output_dir / SUMMARY_PATH.name
    results.to_csv(results_path, index=False)

    run_id = new_run_id("forecast_validation")
    summary = {
        "run_id": run_id,
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "result_rows": len(results),
        "split_summaries": split_summaries,
        "scenarios_held_out": sorted(df["scenario"].dropna().astype(str).unique().tolist()),
        "interpretation": (
            "These are temporal and scenario-held-out baselines. Candidate forecasting models "
            "must be evaluated on the same partitions before claiming operational improvement."
        ),
        "results_path": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_path, manifest_hash = write_run_manifest(
        run_id=run_id,
        run_type="forecast_baseline_validation",
        parameters={"test_fraction": 0.25, "scenario_holdout": True},
        input_paths=[data_path],
        output_paths=[results_path, summary_path],
        metrics={"dataset_rows": len(df), "result_rows": len(results)},
        notes=[
            "No random row split is used.",
            "Results are baseline evidence, not production-model approval.",
        ],
        manifest_dir=manifest_dir,
    )
    return {
        **summary,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
    }


def main() -> None:
    summary = run_forecast_baseline_validation()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
