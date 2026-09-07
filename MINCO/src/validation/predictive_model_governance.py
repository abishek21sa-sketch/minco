"""Time-respecting predictive-model validation and governance for MINCO.

This module deliberately retrains compact candidate models on the governed
forecast dataset. Historical pickle artifacts are not trusted as production
models because their original environments and random-split validation do not
satisfy the MINCO validation standard.
"""
from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.paths import PROJECT_ROOT, RESULTS_DIR
from src.validation.run_manifest import (
    RUN_MANIFEST_DIR,
    new_run_id,
    sha256_file,
    write_run_manifest,
)
from src.validation.temporal_splits import (
    chronological_holdout,
    grouped_chronological_holdout,
    leave_one_group_out,
)
from src.version import __version__

DATA_PATH = RESULTS_DIR / "ai_datasets" / "forecast_dataset_combined.csv"
OUTPUT_DIR = RESULTS_DIR / "validation"
MODEL_DIR = RESULTS_DIR / "model_registry" / "models"
RESULTS_PATH = OUTPUT_DIR / "predictive_model_validation.csv"
LEAKAGE_PATH = OUTPUT_DIR / "predictive_feature_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "predictive_model_governance_summary.json"
REGISTRY_PATH = RESULTS_DIR / "model_registry" / "predictive_model_registry.csv"

RANDOM_STATE = 20260730
GROUP_COLUMNS = ("dataset_source", "scenario", "policy_name", "design_name")
IDENTIFIER_COLUMNS = {"replication"}
FUTURE_PREFIXES = ("target_next_",)
FUTURE_EXACT = {"next_regime_label"}


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    target: str
    task_type: str
    persistence_feature: str
    approval_metric: str
    structural_dependency_features: tuple[str, ...] = ()


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_name="next_unsafe_excess",
        target="target_next_total_unsafe_excess",
        task_type="regression",
        persistence_feature="total_unsafe_excess",
        approval_metric="mae",
    ),
    TaskSpec(
        task_name="next_blocked_arrivals",
        target="target_next_total_blocked_arrivals",
        task_type="regression",
        persistence_feature="total_blocked_arrivals",
        approval_metric="mae",
    ),
    TaskSpec(
        task_name="next_utilization_critical",
        target="target_next_utilization_critical_flag",
        task_type="binary_classification",
        persistence_feature="utilization_critical_flag",
        approval_metric="f1_macro",
    ),
    TaskSpec(
        task_name="next_regime",
        target="next_regime_label",
        task_type="multiclass_classification",
        persistence_feature="current_regime",
        approval_metric="f1_macro",
        structural_dependency_features=("current_regime", "current_regime_code"),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def governed_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Return current/past/exogenous features while excluding all future targets."""
    prohibited = {
        column
        for column in df.columns
        if column == target
        or column in FUTURE_EXACT
        or any(column.startswith(prefix) for prefix in FUTURE_PREFIXES)
        or column in IDENTIFIER_COLUMNS
    }
    features = [column for column in df.columns if column not in prohibited]
    if target in features:
        raise AssertionError("Target column entered the governed feature set")
    if any(column.startswith("target_next_") for column in features):
        raise AssertionError("Future target column entered the governed feature set")
    if "next_regime_label" in features:
        raise AssertionError("Future regime label entered the governed feature set")
    return features


def _deterministic_dependency_rate(
    df: pd.DataFrame,
    *,
    feature: str,
    target: str,
) -> float | None:
    valid = df[[feature, target]].dropna()
    if valid.empty or valid[feature].nunique() > 100:
        return None
    mapping_sizes = valid.groupby(feature, dropna=False)[target].nunique(dropna=False)
    deterministic_values = mapping_sizes[mapping_sizes == 1].index
    if len(deterministic_values) == 0:
        return 0.0
    return float(valid[feature].isin(deterministic_values).mean())


def audit_task_features(df: pd.DataFrame, spec: TaskSpec) -> tuple[list[str], list[dict[str, Any]]]:
    features = governed_feature_columns(df, spec.target)
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        if column == spec.target:
            reason = "target_column"
            allowed = False
        elif column in FUTURE_EXACT or any(column.startswith(prefix) for prefix in FUTURE_PREFIXES):
            reason = "future_outcome_column"
            allowed = False
        elif column in IDENTIFIER_COLUMNS:
            reason = "replication_identifier"
            allowed = False
        else:
            reason = "current_past_or_exogenous_feature"
            allowed = True

        dependency_rate = None
        if column in spec.structural_dependency_features and column in df.columns:
            dependency_rate = _deterministic_dependency_rate(df, feature=column, target=spec.target)
            if dependency_rate is not None and dependency_rate >= 0.999:
                reason = "deterministic_target_construction_dependency"

        rows.append(
            {
                "task_name": spec.task_name,
                "target": spec.target,
                "feature": column,
                "allowed_for_training": allowed,
                "audit_reason": reason,
                "deterministic_dependency_rate": dependency_rate,
            }
        )
    return features, rows


def _preprocessor(df: pd.DataFrame, features: Sequence[str], *, scale_numeric: bool) -> ColumnTransformer:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in features if column not in numeric]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def candidate_models(df: pd.DataFrame, features: Sequence[str], spec: TaskSpec) -> dict[str, Pipeline]:
    if spec.task_type == "regression":
        return {
            "ridge": Pipeline(
                [
                    ("preprocess", _preprocessor(df, features, scale_numeric=True)),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
            "hist_gradient_boosting": Pipeline(
                [
                    ("preprocess", _preprocessor(df, features, scale_numeric=False)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            learning_rate=0.06,
                            max_iter=120,
                            max_leaf_nodes=15,
                            l2_regularization=1.0,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        }

    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor(df, features, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", _preprocessor(df, features, scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=120,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _expected_calibration_error(
    actual: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    actual = np.asarray(actual)
    if probabilities.ndim == 1:
        confidence = probabilities
        predicted = (probabilities >= 0.5).astype(int)
    else:
        confidence = probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)
        if not np.issubdtype(actual.dtype, np.number):
            raise ValueError("Multiclass ECE requires encoded numeric labels")

    correct = (predicted == actual).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        weight = float(mask.mean())
        ece += weight * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def _regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    y = actual.to_numpy(dtype=float)
    pred = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def _classification_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    y = actual.astype(str).to_numpy()
    pred = np.asarray(predicted).astype(str)
    probability_array = np.asarray(probabilities, dtype=float)
    class_labels = np.asarray(classes).astype(str)

    encoded = np.array([int(np.where(class_labels == value)[0][0]) for value in y], dtype=int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y, probability_array, labels=class_labels)),
        "ece_10": _expected_calibration_error(encoded, probability_array),
    }
    if len(class_labels) == 2:
        positive_probability = probability_array[:, 1]
        positive_label = class_labels[1]
        binary_actual = (y == positive_label).astype(int)
        metrics["brier_score"] = float(brier_score_loss(binary_actual, positive_probability))
    else:
        one_hot = np.eye(len(class_labels), dtype=float)[encoded]
        metrics["brier_score"] = float(np.mean(np.sum((probability_array - one_hot) ** 2, axis=1)))
    return metrics


def _baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: TaskSpec,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    if spec.task_type == "regression":
        return (
            test[spec.persistence_feature].to_numpy(dtype=float),
            None,
            None,
            "persistence_current",
        )

    persistence = test[spec.persistence_feature].astype(str).to_numpy()
    majority_series = train[spec.target].astype(str).mode(dropna=True)
    if majority_series.empty:
        raise ValueError(f"No training majority class for {spec.task_name}")
    majority = np.repeat(majority_series.iloc[0], len(test))
    actual = test[spec.target].astype(str).to_numpy()
    persistence_f1 = f1_score(actual, persistence, average="macro", zero_division=0)
    majority_f1 = f1_score(actual, majority, average="macro", zero_division=0)
    hard = persistence if persistence_f1 >= majority_f1 else majority
    name = "persistence_current" if persistence_f1 >= majority_f1 else "training_majority_class"

    classes = np.array(sorted(train[spec.target].dropna().astype(str).unique().tolist()))
    priors = train[spec.target].astype(str).value_counts(normalize=True)
    probabilities = np.zeros((len(test), len(classes)), dtype=float)
    for index, label in enumerate(classes):
        probabilities[:, index] = float(priors.get(label, 0.0))
    return hard, probabilities, classes, name


def _evaluate_one(
    df: pd.DataFrame,
    spec: TaskSpec,
    features: Sequence[str],
    train_index: pd.Index,
    test_index: pd.Index,
    *,
    split_strategy: str,
    held_out_group: str | None,
) -> list[dict[str, Any]]:
    train = df.loc[train_index].dropna(subset=[spec.target])
    test = df.loc[test_index].dropna(subset=[spec.target, spec.persistence_feature])
    if train.empty or test.empty:
        return []

    rows: list[dict[str, Any]] = []
    baseline_pred, baseline_prob, baseline_classes, baseline_name = _baseline_predictions(
        train,
        test,
        spec,
    )
    if spec.task_type == "regression":
        baseline_metrics = _regression_metrics(test[spec.target], baseline_pred)
    else:
        assert baseline_prob is not None and baseline_classes is not None
        baseline_metrics = _classification_metrics(
            test[spec.target],
            baseline_pred,
            baseline_prob,
            baseline_classes,
        )
    rows.append(
        {
            "task_name": spec.task_name,
            "target": spec.target,
            "task_type": spec.task_type,
            "split_strategy": split_strategy,
            "held_out_group": held_out_group,
            "model_name": baseline_name,
            "model_role": "baseline",
            "n_train": len(train),
            "n_test": len(test),
            "n_features": len(features),
            **baseline_metrics,
        }
    )

    x_train = train[list(features)]
    x_test = test[list(features)]
    y_train = train[spec.target]
    y_test = test[spec.target]
    for model_name, model in candidate_models(df, features, spec).items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        if spec.task_type == "regression":
            metrics = _regression_metrics(y_test, prediction)
            improvement = (baseline_metrics["mae"] - metrics["mae"]) / max(
                baseline_metrics["mae"],
                1e-12,
            )
        else:
            probabilities = model.predict_proba(x_test)
            classes = np.asarray(model.classes_)
            metrics = _classification_metrics(y_test, prediction, probabilities, classes)
            improvement = metrics["f1_macro"] - baseline_metrics["f1_macro"]

        rows.append(
            {
                "task_name": spec.task_name,
                "target": spec.target,
                "task_type": spec.task_type,
                "split_strategy": split_strategy,
                "held_out_group": held_out_group,
                "model_name": model_name,
                "model_role": "candidate",
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(features),
                "improvement_over_baseline": float(improvement),
                **metrics,
            }
        )
    return rows


def _splits(df: pd.DataFrame) -> list[tuple[str, str | None, pd.Index, pd.Index]]:
    output: list[tuple[str, str | None, pd.Index, pd.Index]] = []
    train, test, meta = chronological_holdout(df, test_fraction=0.25)
    output.append((meta.strategy, None, train, test))

    group_columns = [column for column in GROUP_COLUMNS if column in df.columns]
    train, test, meta = grouped_chronological_holdout(
        df,
        group_cols=group_columns,
        test_fraction=0.25,
    )
    output.append((meta.strategy, None, train, test))

    for scenario in sorted(df["scenario"].dropna().astype(str).unique().tolist()):
        train, test = leave_one_group_out(df, group_col="scenario", held_out_value=scenario)
        output.append(("leave_one_scenario_out", scenario, train, test))
    return output


def _task_governance(
    results: pd.DataFrame,
    feature_audit: pd.DataFrame,
    spec: TaskSpec,
) -> dict[str, Any]:
    task_rows = results[(results["task_name"] == spec.task_name) & (results["model_role"] == "candidate")]
    if task_rows.empty:
        return {
            "task_name": spec.task_name,
            "target": spec.target,
            "status": "rejected",
            "reason": "no_candidate_results",
            "selected_model": None,
        }

    deterministic = feature_audit[
        (feature_audit["task_name"] == spec.task_name)
        & (feature_audit["audit_reason"] == "deterministic_target_construction_dependency")
    ]
    if not deterministic.empty:
        return {
            "task_name": spec.task_name,
            "target": spec.target,
            "status": "rejected",
            "reason": "target_is_deterministically_constructed_from_current_state_labels",
            "selected_model": None,
            "evidence_features": deterministic["feature"].tolist(),
        }

    model_summary = (
        task_rows.groupby("model_name", as_index=False)
        .agg(
            median_improvement=("improvement_over_baseline", "median"),
            minimum_improvement=("improvement_over_baseline", "min"),
            mean_improvement=("improvement_over_baseline", "mean"),
            split_count=("split_strategy", "count"),
        )
        .sort_values(["median_improvement", "minimum_improvement"], ascending=False)
    )
    best = model_summary.iloc[0]
    best_rows = task_rows[task_rows["model_name"] == best["model_name"]]
    temporal = best_rows[best_rows["split_strategy"].isin(
        ["chronological_holdout", "grouped_chronological_holdout"]
    )]
    scenario_rows = best_rows[best_rows["split_strategy"] == "leave_one_scenario_out"]

    if spec.task_type == "regression":
        temporal_ok = bool((temporal["improvement_over_baseline"] >= 0.0).all())
        scenario_median = float(scenario_rows["improvement_over_baseline"].median())
        r2_median = float(best_rows["r2"].median())
        calibration_ok = True
        approved = (
            float(best["median_improvement"]) >= 0.02
            and float(best["minimum_improvement"]) >= 0.0
            and temporal_ok
            and scenario_median >= 0.0
            and r2_median > 0.0
        )
        conditional = float(best["median_improvement"]) > 0.0 and r2_median > 0.0
    else:
        temporal_ok = bool((temporal["improvement_over_baseline"] >= 0.0).all())
        scenario_median = float(scenario_rows["improvement_over_baseline"].median())
        calibration_ok = bool((best_rows["ece_10"] <= 0.15).all())
        approved = (
            float(best["median_improvement"]) >= 0.02
            and float(best["minimum_improvement"]) >= -0.02
            and temporal_ok
            and scenario_median >= 0.0
            and calibration_ok
        )
        conditional = float(best["median_improvement"]) > 0.0 and calibration_ok

    if approved:
        status = "approved_for_reference_case"
        reason = "beats_governed_baselines_across_required_validation_partitions"
    elif conditional:
        status = "conditional_research_only"
        reason = "partial_improvement_without_full_cross_partition_approval"
    else:
        status = "rejected"
        reason = "does_not_reliably_beat_governed_baselines"

    decision = {
        "task_name": spec.task_name,
        "target": spec.target,
        "status": status,
        "reason": reason,
        "selected_model": str(best["model_name"]),
        "median_improvement": float(best["median_improvement"]),
        "minimum_improvement": float(best["minimum_improvement"]),
        "mean_improvement": float(best["mean_improvement"]),
        "temporal_splits_non_degrading": temporal_ok,
        "scenario_holdout_median_improvement": scenario_median,
        "calibration_threshold_met": calibration_ok,
        "split_count": int(best["split_count"]),
    }
    if spec.task_type == "regression":
        decision["median_r2"] = r2_median
    else:
        decision["maximum_ece_10"] = float(best_rows["ece_10"].max())
        decision["median_brier_score"] = float(best_rows["brier_score"].median())
    return decision


def _train_and_register(
    df: pd.DataFrame,
    feature_map: dict[str, list[str]],
    governance: list[dict[str, Any]],
    *,
    data_path: Path,
    model_dir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    data_hash = sha256_file(data_path)
    registry_rows: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    specs = {spec.task_name: spec for spec in TASKS}

    for decision in governance:
        spec = specs[decision["task_name"]]
        model_name = decision.get("selected_model")
        status = decision["status"]
        model_path: Path | None = None
        model_hash: str | None = None

        if model_name and status != "rejected":
            valid = df.dropna(subset=[spec.target])
            features = feature_map[spec.task_name]
            model = candidate_models(df, features, spec)[model_name]
            model.fit(valid[features], valid[spec.target])
            model_path = model_dir / f"{spec.task_name}__{model_name}.joblib"
            joblib.dump(model, model_path, compress=3)
            model_hash = sha256_file(model_path)
            artifact_paths.append(model_path)

        registry_rows.append(
            {
                "registered_at": _utc_now(),
                "task_name": spec.task_name,
                "target": spec.target,
                "task_type": spec.task_type,
                "governance_status": status,
                "selected_model": model_name,
                "model_path": _relative(model_path) if model_path else None,
                "model_sha256": model_hash,
                "training_dataset_path": _relative(data_path),
                "training_dataset_sha256": data_hash,
                "training_rows": int(df[spec.target].notna().sum()),
                "feature_count": len(feature_map[spec.task_name]),
                "median_improvement": decision.get("median_improvement"),
                "minimum_improvement": decision.get("minimum_improvement"),
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "minco_version": __version__,
                "deployment_scope": (
                    "synthetic_reference_case_only"
                    if status == "approved_for_reference_case"
                    else "research_only"
                    if status == "conditional_research_only"
                    else "not_deployable"
                ),
            }
        )

    return pd.DataFrame(registry_rows), artifact_paths


def run_predictive_model_governance(
    *,
    data_path: Path = DATA_PATH,
    output_dir: Path = OUTPUT_DIR,
    model_dir: Path = MODEL_DIR,
    registry_path: Path = REGISTRY_PATH,
    manifest_dir: Path = RUN_MANIFEST_DIR,
) -> dict[str, Any]:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    registry_path = Path(registry_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Governed forecast dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required = {"time_index", "scenario"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Forecast dataset missing required columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    feature_map: dict[str, list[str]] = {}
    feature_audit_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    splits = _splits(df)

    for spec in TASKS:
        features, audit_rows = audit_task_features(df, spec)
        feature_map[spec.task_name] = features
        feature_audit_rows.extend(audit_rows)
        for strategy, held_out, train_index, test_index in splits:
            result_rows.extend(
                _evaluate_one(
                    df,
                    spec,
                    features,
                    train_index,
                    test_index,
                    split_strategy=strategy,
                    held_out_group=held_out,
                )
            )

    results = pd.DataFrame(result_rows)
    feature_audit = pd.DataFrame(feature_audit_rows)
    if results.empty:
        raise RuntimeError("Predictive governance produced no validation rows")

    governance = [
        _task_governance(results, feature_audit, spec)
        for spec in TASKS
    ]
    registry, model_artifacts = _train_and_register(
        df,
        feature_map,
        governance,
        data_path=data_path,
        model_dir=model_dir,
    )

    results_path = output_dir / RESULTS_PATH.name
    leakage_path = output_dir / LEAKAGE_PATH.name
    summary_path = output_dir / SUMMARY_PATH.name
    results.to_csv(results_path, index=False)
    feature_audit.to_csv(leakage_path, index=False)
    registry.to_csv(registry_path, index=False)

    status_counts = pd.Series([item["status"] for item in governance]).value_counts().to_dict()
    run_id = new_run_id("predictive_governance")
    summary = {
        "run_id": run_id,
        "status": "passed",
        "method": "time_respecting_and_scenario_held_out_model_governance",
        "dataset": {
            "path": _relative(data_path),
            "rows": len(df),
            "columns": len(df.columns),
            "sha256": sha256_file(data_path),
        },
        "validation": {
            "split_count": len(splits),
            "strategies": sorted(results["split_strategy"].unique().tolist()),
            "result_rows": len(results),
            "candidate_result_rows": int((results["model_role"] == "candidate").sum()),
            "baseline_result_rows": int((results["model_role"] == "baseline").sum()),
        },
        "task_decisions": governance,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "claim_boundary": (
            "Approvals apply only to the bundled synthetic Meridian reference case. "
            "No model is approved for real-hospital deployment or clinical decision-making."
        ),
        "outputs": {
            "validation_results": _relative(results_path),
            "feature_audit": _relative(leakage_path),
            "model_registry": _relative(registry_path),
            "model_artifacts": [_relative(path) for path in model_artifacts],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The summary is written again after the manifest path/hash are known, so it
    # is intentionally not listed as an immutable manifest output.
    output_paths: list[Path] = [results_path, leakage_path, registry_path]
    output_paths.extend(model_artifacts)
    manifest_path, manifest_hash = write_run_manifest(
        run_id=run_id,
        run_type="predictive_model_governance",
        parameters={
            "random_state": RANDOM_STATE,
            "test_fraction": 0.25,
            "scenario_holdout": True,
            "candidate_models": [
                "ridge",
                "logistic_regression",
                "hist_gradient_boosting",
            ],
        },
        input_paths=[data_path],
        output_paths=output_paths,
        metrics={
            "task_count": len(TASKS),
            "validation_result_rows": len(results),
            "status_counts": summary["status_counts"],
        },
        model_info={
            "registry_path": _relative(registry_path),
            "artifact_count": len(model_artifacts),
        },
        notes=[
            "No random row split is used.",
            "All future target columns and replication identifiers are excluded.",
            "The next-regime task is rejected when its label is deterministically constructed from current-state labels.",
            "Approvals are synthetic-reference-case approvals only.",
        ],
        manifest_dir=manifest_dir,
    )
    summary["manifest_path"] = str(manifest_path)
    summary["manifest_sha256"] = manifest_hash
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(run_predictive_model_governance(), indent=2))


if __name__ == "__main__":
    main()
