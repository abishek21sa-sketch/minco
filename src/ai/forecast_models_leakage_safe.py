from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RESULTS_DIR = Path("results")
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
AI_MODELS_DIR = RESULTS_DIR / "ai_models"
AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"

AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
AI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = AI_DATA_DIR / "forecast_dataset_combined.csv"

RESULTS_PATH = AI_REPORTS_DIR / "forecast_model_results_leakage_safe.csv"
REGISTRY_PATH = AI_REPORTS_DIR / "forecast_model_registry_leakage_safe.csv"
SUMMARY_PATH = AI_REPORTS_DIR / "forecast_model_summary_leakage_safe.json"


# ============================================================
# Leakage rules
# ============================================================

TARGET_COLS = [
    "target_next_total_unsafe_excess",
    "target_next_total_blocked_arrivals",
    "target_next_utilization_critical_flag",
    "next_regime_label",
]

FUTURE_OR_TARGET_PREFIXES = [
    "target_next_",
    "next_",
]

DIRECT_REGIME_LEAKAGE_COLS = [
    "current_regime",
    "current_regime_code",
]

ALLOWED_CATEGORICAL_BASE = [
    "policy_name",
    "scenario",
    "design_name",
    "dataset_source",
]

BASE_NUMERIC_FEATURES = [
    "time_index",
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


# ============================================================
# IO
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def save_pickle(model: Pipeline, filepath: Path) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


# ============================================================
# Feature engineering
# ============================================================

def is_leakage_column(col: str) -> bool:
    if col in TARGET_COLS:
        return True

    if col in DIRECT_REGIME_LEAKAGE_COLS:
        return True

    for prefix in FUTURE_OR_TARGET_PREFIXES:
        if col.startswith(prefix):
            return True

    return False


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in BASE_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    eps = 1e-6

    out["utilization_squared"] = out["max_utilization_ratio"] ** 2
    out["blocked_x_utilization"] = out["total_blocked_arrivals"] * out["max_utilization_ratio"]
    out["unsafe_x_utilization"] = out["total_unsafe_excess"] * out["max_utilization_ratio"]
    out["overflow_x_utilization"] = out["total_overflow_excess"] * out["max_utilization_ratio"]

    out["blocked_to_unsafe_ratio"] = out["total_blocked_arrivals"] / (
        out["total_unsafe_excess"].abs() + eps
    )
    out["overflow_to_unsafe_ratio"] = out["total_overflow_excess"] / (
        out["total_unsafe_excess"].abs() + eps
    )

    out["unsafe_momentum_1"] = out["total_unsafe_excess"] - out["total_unsafe_excess_lag1"]
    out["unsafe_momentum_2"] = out["total_unsafe_excess_lag1"] - out["total_unsafe_excess_lag2"]

    out["blocked_momentum_1"] = out["total_blocked_arrivals"] - out["total_blocked_arrivals_lag1"]
    out["blocked_momentum_2"] = out["total_blocked_arrivals_lag1"] - out["total_blocked_arrivals_lag2"]

    out["utilization_momentum_1"] = out["max_utilization_ratio"] - out["max_utilization_ratio_lag1"]
    out["overflow_momentum_1"] = out["total_overflow_excess"] - out["total_overflow_excess_lag1"]
    out["surge_gap_momentum_1"] = out["total_surge_gap"] - out["total_surge_gap_lag1"]

    out["unsafe_pressure_index"] = (
        out["total_unsafe_excess"]
        + out["total_overflow_excess"]
        + out["num_unsafe_rows"]
        + 5.0 * np.maximum(out["max_utilization_ratio"] - 1.0, 0.0)
    )

    out["access_pressure_index"] = (
        out["total_blocked_arrivals"]
        + out["total_surge_gap"]
        + 2.0 * out["blocked_high_flag"]
    )

    out["combined_stress_index"] = (
        out["unsafe_pressure_index"]
        + 0.5 * out["access_pressure_index"]
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def choose_leakage_safe_features(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[list[str], list[str]]:
    engineered_numeric = [
        "utilization_squared",
        "blocked_x_utilization",
        "unsafe_x_utilization",
        "overflow_x_utilization",
        "blocked_to_unsafe_ratio",
        "overflow_to_unsafe_ratio",
        "unsafe_momentum_1",
        "unsafe_momentum_2",
        "blocked_momentum_1",
        "blocked_momentum_2",
        "utilization_momentum_1",
        "overflow_momentum_1",
        "surge_gap_momentum_1",
        "unsafe_pressure_index",
        "access_pressure_index",
        "combined_stress_index",
    ]

    candidate_numeric = BASE_NUMERIC_FEATURES + engineered_numeric
    candidate_categorical = ALLOWED_CATEGORICAL_BASE

    numeric_features = []
    for c in candidate_numeric:
        if c in df.columns and c != target_col and not is_leakage_column(c):
            numeric_features.append(c)

    categorical_features = []
    for c in candidate_categorical:
        if c in df.columns and c != target_col and not is_leakage_column(c):
            categorical_features.append(c)

    return numeric_features, categorical_features


def assert_no_leakage(
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
) -> None:
    all_features = numeric_features + categorical_features

    bad = []
    for col in all_features:
        if col == target_col or is_leakage_column(col):
            bad.append(col)

    if bad:
        raise ValueError(
            f"Leakage-safe feature check failed. Bad columns: {bad}"
        )


# ============================================================
# Preprocessors / models
# ============================================================

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    if scale_numeric:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def build_regression_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    linear_pre = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=True,
    )

    tree_pre = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
    )

    return {
        "linear_regression_leakage_safe": Pipeline(
            steps=[
                ("preprocess", linear_pre),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_leakage_safe": Pipeline(
            steps=[
                ("preprocess", linear_pre),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest_leakage_safe": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "extra_trees_leakage_safe": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "gradient_boosting_leakage_safe": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.04,
                    max_depth=3,
                    min_samples_leaf=3,
                    random_state=42,
                )),
            ]
        ),
    }


def build_classification_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    linear_pre = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=True,
    )

    tree_pre = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
    )

    return {
        "logistic_regression_leakage_safe": Pipeline(
            steps=[
                ("preprocess", linear_pre),
                ("model", LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                )),
            ]
        ),
        "random_forest_classifier_leakage_safe": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "extra_trees_classifier_leakage_safe": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", ExtraTreesClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        ),
        "gradient_boosting_classifier_leakage_safe": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                ("model", GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.04,
                    max_depth=3,
                    random_state=42,
                )),
            ]
        ),
    }


# ============================================================
# Metrics
# ============================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ============================================================
# Training
# ============================================================

def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    work = df.dropna(subset=[target_col]).copy()

    numeric_features, categorical_features = choose_leakage_safe_features(
        work,
        target_col=target_col,
    )

    assert_no_leakage(numeric_features, categorical_features, target_col)

    feature_cols = numeric_features + categorical_features

    if not feature_cols:
        raise ValueError(f"No leakage-safe feature columns found for target: {target_col}")

    X = work[feature_cols].copy()
    y = work[target_col].copy()

    return X, y, numeric_features, categorical_features


def train_regression_task(
    df: pd.DataFrame,
    target_col: str,
    task_name: str,
) -> tuple[pd.DataFrame, list[dict]]:
    X, y, numeric_features, categorical_features = prepare_xy(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.astype(float),
        test_size=0.25,
        random_state=42,
    )

    rows = []
    registry_rows = []

    models = build_regression_models(numeric_features, categorical_features)

    for model_name, model in models.items():
        print(f"\nTraining {task_name} / {model_name}...", flush=True)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = regression_metrics(y_test.to_numpy(dtype=float), preds)

        model_path = AI_MODELS_DIR / f"{task_name}__{model_name}.pkl"
        save_pickle(model, model_path)

        rows.append(
            {
                "task_name": task_name,
                "target_col": target_col,
                "model_name": model_name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                **metrics,
                "leakage_safe": True,
            }
        )

        registry_rows.append(
            {
                "task_name": task_name,
                "target_col": target_col,
                "model_name": model_name,
                "model_path": str(model_path),
                "leakage_safe": True,
            }
        )

        print(
            f"{model_name}: r2={metrics['r2']:.3f}, mae={metrics['mae']:.3f}, rmse={metrics['rmse']:.3f}",
            flush=True,
        )

    return pd.DataFrame(rows), registry_rows


def train_classification_task(
    df: pd.DataFrame,
    target_col: str,
    task_name: str,
) -> tuple[pd.DataFrame, list[dict]]:
    X, y, numeric_features, categorical_features = prepare_xy(df, target_col)

    y = y.astype(str)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    rows = []
    registry_rows = []

    models = build_classification_models(numeric_features, categorical_features)

    for model_name, model in models.items():
        print(f"\nTraining {task_name} / {model_name}...", flush=True)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = classification_metrics(y_test.to_numpy(), preds)

        model_path = AI_MODELS_DIR / f"{task_name}__{model_name}.pkl"
        save_pickle(model, model_path)

        rows.append(
            {
                "task_name": task_name,
                "target_col": target_col,
                "model_name": model_name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_classes": int(y.nunique()),
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                **metrics,
                "leakage_safe": True,
            }
        )

        registry_rows.append(
            {
                "task_name": task_name,
                "target_col": target_col,
                "model_name": model_name,
                "model_path": str(model_path),
                "leakage_safe": True,
            }
        )

        report_path = AI_REPORTS_DIR / f"{task_name}__{model_name}__classification_report_leakage_safe.txt"
        report_path.write_text(
            classification_report(y_test, preds, zero_division=0),
            encoding="utf-8",
        )

        print(
            f"{model_name}: acc={metrics['accuracy']:.3f}, "
            f"f1_macro={metrics['f1_macro']:.3f}, "
            f"f1_weighted={metrics['f1_weighted']:.3f}",
            flush=True,
        )

    return pd.DataFrame(rows), registry_rows


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nLEAKAGE-SAFE FORECAST MODELS")
    print("=" * 70)

    raw_df = safe_read_csv(DATA_PATH)
    df = add_engineered_features(raw_df)

    regression_tasks = [
        ("target_next_total_unsafe_excess", "forecast_next_unsafe_excess_leakage_safe"),
        ("target_next_total_blocked_arrivals", "forecast_next_blocked_arrivals_leakage_safe"),
    ]

    classification_tasks = [
        ("target_next_utilization_critical_flag", "forecast_next_utilization_critical_leakage_safe"),
        ("next_regime_label", "forecast_next_regime_leakage_safe"),
    ]

    all_results = []
    registry_rows = []

    for target_col, task_name in regression_tasks:
        if target_col not in df.columns:
            print(f"Skipping {task_name}: missing {target_col}")
            continue

        result_df, reg_rows = train_regression_task(
            df=df,
            target_col=target_col,
            task_name=task_name,
        )

        all_results.append(result_df)
        registry_rows.extend(reg_rows)

    for target_col, task_name in classification_tasks:
        if target_col not in df.columns:
            print(f"Skipping {task_name}: missing {target_col}")
            continue

        result_df, reg_rows = train_classification_task(
            df=df,
            target_col=target_col,
            task_name=task_name,
        )

        all_results.append(result_df)
        registry_rows.extend(reg_rows)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    registry_df = pd.DataFrame(registry_rows)

    results_df.to_csv(RESULTS_PATH, index=False)
    registry_df.to_csv(REGISTRY_PATH, index=False)

    best_by_task = []
    if not results_df.empty:
        for task in sorted(results_df["task_name"].unique()):
            tdf = results_df[results_df["task_name"] == task].copy()

            if "r2" in tdf.columns and tdf["r2"].notna().any():
                best = tdf.sort_values("r2", ascending=False).iloc[0]
                score_metric = "r2"
                score_value = float(best["r2"])
            elif "f1_weighted" in tdf.columns and tdf["f1_weighted"].notna().any():
                best = tdf.sort_values("f1_weighted", ascending=False).iloc[0]
                score_metric = "f1_weighted"
                score_value = float(best["f1_weighted"])
            else:
                best = tdf.iloc[0]
                score_metric = "unknown"
                score_value = None

            best_by_task.append(
                {
                    "task_name": task,
                    "best_model": best["model_name"],
                    "score_metric": score_metric,
                    "score_value": score_value,
                }
            )

    summary = {
        "input_path": str(DATA_PATH),
        "n_rows_input": int(len(raw_df)),
        "n_results_rows": int(len(results_df)),
        "n_saved_models": int(len(registry_df)),
        "leakage_rules": {
            "removed_target_prefixes": FUTURE_OR_TARGET_PREFIXES,
            "removed_direct_regime_features": DIRECT_REGIME_LEAKAGE_COLS,
            "allowed_categorical_features": ALLOWED_CATEGORICAL_BASE,
        },
        "best_by_task": best_by_task,
        "results_path": str(RESULTS_PATH),
        "registry_path": str(REGISTRY_PATH),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Leakage-safe results ===")
    print(results_df.to_string(index=False))

    print("\n=== Best by task ===")
    print(pd.DataFrame(best_by_task).to_string(index=False))

    print("\nSaved:")
    print(RESULTS_PATH)
    print(REGISTRY_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()