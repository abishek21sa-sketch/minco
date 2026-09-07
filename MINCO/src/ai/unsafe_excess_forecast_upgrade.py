from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RESULTS_DIR = Path("results")
AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"
AI_MODELS_DIR = RESULTS_DIR / "ai_models"
FIGURES_DIR = RESULTS_DIR / "figures"

AI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = AI_DATA_DIR / "forecast_dataset_combined.csv"

TARGET_COL = "target_next_total_unsafe_excess"

OUTPUT_RESULTS_PATH = AI_REPORTS_DIR / "unsafe_forecast_upgrade_results.csv"
OUTPUT_FEATURE_IMPORTANCE_PATH = AI_REPORTS_DIR / "unsafe_forecast_feature_importance.csv"
OUTPUT_SUMMARY_PATH = AI_REPORTS_DIR / "unsafe_forecast_upgrade_summary.json"

ACTUAL_PRED_FIG_PATH = FIGURES_DIR / "unsafe_forecast_actual_vs_predicted.png"
FEATURE_IMPORTANCE_FIG_PATH = FIGURES_DIR / "unsafe_forecast_feature_importance.png"


BASE_NUMERIC_FEATURES = [
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

BASE_CATEGORICAL_FEATURES = [
    "policy_name",
    "scenario",
    "design_name",
    "dataset_source",
    "current_regime",
]


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def add_unsafe_excess_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in BASE_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in BASE_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].astype(str).fillna("unknown")

    eps = 1e-6

    out["utilization_squared"] = out["max_utilization_ratio"] ** 2
    out["blocked_x_utilization"] = out["total_blocked_arrivals"] * out["max_utilization_ratio"]
    out["unsafe_x_utilization"] = out["total_unsafe_excess"] * out["max_utilization_ratio"]
    out["overflow_x_utilization"] = out["total_overflow_excess"] * out["max_utilization_ratio"]
    out["unsafe_x_regime"] = out["total_unsafe_excess"] * out["current_regime_code"]
    out["blocked_x_regime"] = out["total_blocked_arrivals"] * out["current_regime_code"]
    out["overflow_x_regime"] = out["total_overflow_excess"] * out["current_regime_code"]

    out["blocked_to_unsafe_ratio"] = out["total_blocked_arrivals"] / (
        out["total_unsafe_excess"].abs() + eps
    )
    out["overflow_to_unsafe_ratio"] = out["total_overflow_excess"] / (
        out["total_unsafe_excess"].abs() + eps
    )

    out["unsafe_momentum_1"] = out["total_unsafe_excess"] - out["total_unsafe_excess_lag1"]
    out["unsafe_momentum_2"] = out["total_unsafe_excess_lag1"] - out["total_unsafe_excess_lag2"]
    out["blocked_momentum_1"] = out["total_blocked_arrivals"] - out["total_blocked_arrivals_lag1"]
    out["utilization_momentum_1"] = out["max_utilization_ratio"] - out["max_utilization_ratio_lag1"]
    out["overflow_momentum_1"] = out["total_overflow_excess"] - out["total_overflow_excess_lag1"]

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
        + 2.0 * out["current_regime_code"]
    )

    out["surge_interaction_index"] = (
        out["current_regime_code"]
        * (
            out["total_unsafe_excess"]
            + out["total_blocked_arrivals"]
            + out["total_overflow_excess"]
        )
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    engineered_numeric = [
        "utilization_squared",
        "blocked_x_utilization",
        "unsafe_x_utilization",
        "overflow_x_utilization",
        "unsafe_x_regime",
        "blocked_x_regime",
        "overflow_x_regime",
        "blocked_to_unsafe_ratio",
        "overflow_to_unsafe_ratio",
        "unsafe_momentum_1",
        "unsafe_momentum_2",
        "blocked_momentum_1",
        "utilization_momentum_1",
        "overflow_momentum_1",
        "unsafe_pressure_index",
        "access_pressure_index",
        "combined_stress_index",
        "surge_interaction_index",
    ]

    numeric_features = [
        c for c in BASE_NUMERIC_FEATURES + engineered_numeric
        if c in df.columns and c != TARGET_COL
    ]

    categorical_features = [
        c for c in BASE_CATEGORICAL_FEATURES
        if c in df.columns
    ]

    return numeric_features, categorical_features


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


def build_models(
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
        "linear_regression_engineered": Pipeline(
            steps=[
                ("preprocess", linear_pre),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_engineered": Pipeline(
            steps=[
                ("preprocess", linear_pre),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest_engineered": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_engineered": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting_engineered": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=350,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    denom = np.abs(y_true) + 1e-6
    mape_like = float(np.mean(np.abs(y_true - y_pred) / denom))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_like": mape_like,
    }


def extract_feature_names(
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    pre = model.named_steps["preprocess"]

    names = []

    names.extend(numeric_features)

    try:
        onehot = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_names = onehot.get_feature_names_out(categorical_features).tolist()
        names.extend(cat_names)
    except Exception:
        names.extend(categorical_features)

    return names


def extract_feature_importance(
    model_name: str,
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    reg = model.named_steps["model"]

    if not hasattr(reg, "feature_importances_"):
        return pd.DataFrame()

    feature_names = extract_feature_names(
        model=model,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    importances = reg.feature_importances_

    n = min(len(feature_names), len(importances))

    out = pd.DataFrame(
        {
            "model_name": model_name,
            "feature": feature_names[:n],
            "importance": importances[:n],
        }
    )

    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))

    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    plt.xlabel("Actual next unsafe excess")
    plt.ylabel("Predicted next unsafe excess")
    plt.title(f"Unsafe Excess Forecast: Actual vs Predicted\nBest model: {model_name}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    if importance_df.empty:
        return

    top = importance_df.head(top_n).copy()
    top = top.sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances for Unsafe Excess Forecast")
    plt.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("\nUNSAFE EXCESS FORECAST UPGRADE")
    print("=" * 70)

    raw_df = safe_read_csv(INPUT_PATH)

    if TARGET_COL not in raw_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df = add_unsafe_excess_features(raw_df)

    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()

    numeric_features, categorical_features = get_feature_lists(df)

    X = df[numeric_features + categorical_features].copy()
    y = df[TARGET_COL].astype(float).to_numpy()

    stratify_col = None
    if "current_regime" in df.columns:
        counts = df["current_regime"].value_counts()
        if counts.min() >= 2:
            stratify_col = df["current_regime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_col,
    )

    models = build_models(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    rows = []
    fitted_models: dict[str, Pipeline] = {}
    importance_frames = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...", flush=True)

        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        train_metrics = regression_metrics(y_train, pred_train)
        test_metrics = regression_metrics(y_test, pred_test)

        model_path = AI_MODELS_DIR / f"unsafe_forecast_{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        rows.append(
            {
                "task_name": "forecast_next_unsafe_excess_upgraded",
                "target_col": TARGET_COL,
                "model_name": model_name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_r2": train_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
                "test_mape_like": test_metrics["mape_like"],
                "model_path": str(model_path),
            }
        )

        fitted_models[model_name] = model

        imp = extract_feature_importance(
            model_name=model_name,
            model=model,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        if not imp.empty:
            importance_frames.append(imp)

        print(
            f"{model_name}: test_r2={test_metrics['r2']:.3f}, "
            f"test_mae={test_metrics['mae']:.3f}, "
            f"test_rmse={test_metrics['rmse']:.3f}",
            flush=True,
        )

    results_df = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_RESULTS_PATH, index=False)

    best_row = results_df.iloc[0]
    best_model_name = str(best_row["model_name"])
    best_model = fitted_models[best_model_name]
    best_pred = best_model.predict(X_test)

    if importance_frames:
        all_importance = pd.concat(importance_frames, ignore_index=True)
        all_importance.to_csv(OUTPUT_FEATURE_IMPORTANCE_PATH, index=False)

        best_importance = all_importance[
            all_importance["model_name"] == best_model_name
        ].copy()
        plot_feature_importance(best_importance, FEATURE_IMPORTANCE_FIG_PATH)
    else:
        all_importance = pd.DataFrame()
        all_importance.to_csv(OUTPUT_FEATURE_IMPORTANCE_PATH, index=False)

    plot_actual_vs_predicted(
        y_true=y_test,
        y_pred=best_pred,
        model_name=best_model_name,
        path=ACTUAL_PRED_FIG_PATH,
    )

    summary = {
        "input_path": str(INPUT_PATH),
        "target_col": TARGET_COL,
        "n_rows_raw": int(len(raw_df)),
        "n_rows_used": int(len(df)),
        "n_numeric_features": int(len(numeric_features)),
        "n_categorical_features": int(len(categorical_features)),
        "best_model": best_model_name,
        "best_test_r2": float(best_row["test_r2"]),
        "best_test_mae": float(best_row["test_mae"]),
        "best_test_rmse": float(best_row["test_rmse"]),
        "results_path": str(OUTPUT_RESULTS_PATH),
        "feature_importance_path": str(OUTPUT_FEATURE_IMPORTANCE_PATH),
        "actual_vs_predicted_figure": str(ACTUAL_PRED_FIG_PATH),
        "feature_importance_figure": str(FEATURE_IMPORTANCE_FIG_PATH),
    }

    OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Unsafe forecast upgrade results ===")
    print(results_df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_RESULTS_PATH)
    print(OUTPUT_FEATURE_IMPORTANCE_PATH)
    print(OUTPUT_SUMMARY_PATH)
    print(ACTUAL_PRED_FIG_PATH)
    print(FEATURE_IMPORTANCE_FIG_PATH)


if __name__ == "__main__":
    main()