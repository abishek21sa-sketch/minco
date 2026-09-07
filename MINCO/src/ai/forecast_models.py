from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RESULTS_DIR = Path("results")
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
AI_MODELS_DIR = RESULTS_DIR / "ai_models"
AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"

AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
AI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = AI_DATA_DIR / "forecast_dataset_combined.csv"


# ============================================================
# IO
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


# ============================================================
# Feature selection
# ============================================================

def choose_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Returns:
        numeric_features, categorical_features
    """
    candidate_numeric = [
        "time_index",
        "current_regime_code",
        "lookahead_horizon",
        "icu_capacity_scale",
        "transfer_capacity_scale",
        "demand_scale",

        "total_unsafe_excess",
        "total_blocked_arrivals",
        "max_utilization_ratio",
        "num_unsafe_rows",
        "total_overflow_excess",
        "total_surge_gap",

        "total_unsafe_excess_lag1",
        "total_unsafe_excess_lag2",
        "total_unsafe_excess_rollmean_3",
        "total_unsafe_excess_rollmax_3",

        "total_blocked_arrivals_lag1",
        "total_blocked_arrivals_lag2",
        "total_blocked_arrivals_rollmean_3",
        "total_blocked_arrivals_rollmax_3",

        "max_utilization_ratio_lag1",
        "max_utilization_ratio_lag2",
        "max_utilization_ratio_rollmean_3",
        "max_utilization_ratio_rollmax_3",

        "num_unsafe_rows_lag1",
        "num_unsafe_rows_lag2",
        "num_unsafe_rows_rollmean_3",
        "num_unsafe_rows_rollmax_3",

        "total_overflow_excess_lag1",
        "total_overflow_excess_lag2",
        "total_overflow_excess_rollmean_3",
        "total_overflow_excess_rollmax_3",

        "total_surge_gap_lag1",
        "total_surge_gap_lag2",
        "total_surge_gap_rollmean_3",
        "total_surge_gap_rollmax_3",

        "unsafe_positive_flag",
        "unsafe_high_flag",
        "blocked_positive_flag",
        "blocked_high_flag",
        "utilization_high_flag",
        "utilization_critical_flag",
        "overflow_positive_flag",
        "unsafe_rows_positive_flag",
    ]

    candidate_categorical = [
        "policy_name",
        "scenario",
        "design_name",
        "dataset_source",
        "current_regime",
    ]

    numeric_features = [c for c in candidate_numeric if c in df.columns]
    categorical_features = [c for c in candidate_categorical if c in df.columns]

    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# ============================================================
# Model builders
# ============================================================

def build_regression_models(
    preprocessor: ColumnTransformer,
) -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest_regressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
    }


def build_classification_models(
    preprocessor: ColumnTransformer,
) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                )),
            ]
        ),
        "random_forest_classifier": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
    }


# ============================================================
# Evaluation
# ============================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# ============================================================
# Training tasks
# ============================================================

def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    work = df.copy()
    work = work.dropna(subset=[target_col]).copy()

    numeric_features, categorical_features = choose_feature_columns(work)
    feature_cols = numeric_features + categorical_features

    X = work[feature_cols].copy()
    y = work[target_col].copy()

    return X, y, numeric_features, categorical_features


def train_regression_task(
    df: pd.DataFrame,
    target_col: str,
    task_name: str,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    X, y, numeric_features, categorical_features = prepare_xy(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(float),
        test_size=0.25,
        random_state=42,
    )

    results = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in build_regression_models(
        build_preprocessor(numeric_features, categorical_features)
    ).items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = regression_metrics(y_test.to_numpy(), preds)

        row = {
            "task_name": task_name,
            "target_col": target_col,
            "model_name": model_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            **metrics,
        }
        results.append(row)
        fitted_models[model_name] = model

    return pd.DataFrame(results), fitted_models


def train_classification_task(
    df: pd.DataFrame,
    target_col: str,
    task_name: str,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, str]]:
    X, y, numeric_features, categorical_features = prepare_xy(df, target_col)

    y = y.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    results = []
    fitted_models: dict[str, Pipeline] = {}
    reports: dict[str, str] = {}

    for model_name, model in build_classification_models(
        build_preprocessor(numeric_features, categorical_features)
    ).items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = classification_metrics(y_test.to_numpy(), preds)

        row = {
            "task_name": task_name,
            "target_col": target_col,
            "model_name": model_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": int(pd.Series(y).nunique()),
            **metrics,
        }
        results.append(row)
        fitted_models[model_name] = model
        reports[model_name] = classification_report(y_test, preds, zero_division=0)

    return pd.DataFrame(results), fitted_models, reports


# ============================================================
# Model saving
# ============================================================

def save_model(model: Pipeline, filepath: Path) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def save_text(text: str, filepath: Path) -> None:
    filepath.write_text(text, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    df = safe_read_csv(DATA_PATH)

    regression_tasks = [
        ("target_next_total_unsafe_excess", "forecast_next_unsafe_excess"),
        ("target_next_total_blocked_arrivals", "forecast_next_blocked_arrivals"),
    ]

    classification_tasks = [
        ("target_next_utilization_critical_flag", "forecast_next_utilization_critical"),
        ("next_regime_label", "forecast_next_regime"),
    ]

    all_results = []
    saved_model_rows = []

    print("\nFORECAST MODELS")
    print("=" * 60)

    # ----------------------------
    # Regression tasks
    # ----------------------------
    for target_col, task_name in regression_tasks:
        if target_col not in df.columns:
            print(f"\nSkipping regression task {task_name}: missing {target_col}")
            continue

        result_df, fitted_models = train_regression_task(
            df=df,
            target_col=target_col,
            task_name=task_name,
        )

        print(f"\nRegression task: {task_name}")
        print(result_df)
        all_results.append(result_df)

        for model_name, model in fitted_models.items():
            model_path = AI_MODELS_DIR / f"{task_name}__{model_name}.pkl"
            save_model(model, model_path)

            saved_model_rows.append(
                {
                    "task_name": task_name,
                    "target_col": target_col,
                    "model_name": model_name,
                    "model_path": str(model_path),
                }
            )

    # ----------------------------
    # Classification tasks
    # ----------------------------
    for target_col, task_name in classification_tasks:
        if target_col not in df.columns:
            print(f"\nSkipping classification task {task_name}: missing {target_col}")
            continue

        result_df, fitted_models, reports = train_classification_task(
            df=df,
            target_col=target_col,
            task_name=task_name,
        )

        print(f"\nClassification task: {task_name}")
        print(result_df)
        all_results.append(result_df)

        for model_name, model in fitted_models.items():
            model_path = AI_MODELS_DIR / f"{task_name}__{model_name}.pkl"
            save_model(model, model_path)

            saved_model_rows.append(
                {
                    "task_name": task_name,
                    "target_col": target_col,
                    "model_name": model_name,
                    "model_path": str(model_path),
                }
            )

            report_path = AI_REPORTS_DIR / f"{task_name}__{model_name}__classification_report.txt"
            save_text(reports[model_name], report_path)

    # ----------------------------
    # Save result tables
    # ----------------------------
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
    else:
        results_df = pd.DataFrame()

    model_registry_df = pd.DataFrame(saved_model_rows)

    results_path = AI_REPORTS_DIR / "forecast_model_results.csv"
    registry_path = AI_REPORTS_DIR / "forecast_model_registry.csv"
    summary_path = AI_REPORTS_DIR / "forecast_model_summary.json"

    if not results_df.empty:
        results_df.to_csv(results_path, index=False)

    if not model_registry_df.empty:
        model_registry_df.to_csv(registry_path, index=False)

    summary = {
        "n_rows_input": int(len(df)),
        "n_results_rows": int(len(results_df)),
        "n_saved_models": int(len(model_registry_df)),
        "tasks_run": sorted(results_df["task_name"].unique().tolist()) if not results_df.empty else [],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(results_path if results_path.exists() else f"{results_path} [not created]")
    print(registry_path if registry_path.exists() else f"{registry_path} [not created]")
    print(summary_path)

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()