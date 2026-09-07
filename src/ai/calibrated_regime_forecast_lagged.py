from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")

AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"
AI_MODELS_DIR = RESULTS_DIR / "ai_models"

AI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = RESULTS_DIR / "tables" / "regime_recalibration_row_level.csv"

RESULTS_OUT = AI_REPORTS_DIR / "calibrated_regime_lagged_results.csv"
CONFUSION_OUT = AI_REPORTS_DIR / "calibrated_regime_lagged_confusion_matrix.csv"
SUMMARY_OUT = AI_REPORTS_DIR / "calibrated_regime_lagged_summary.json"
REPORT_OUT = AI_REPORTS_DIR / "calibrated_regime_lagged_report.txt"
FEATURE_IMPORTANCE_OUT = AI_REPORTS_DIR / "calibrated_regime_lagged_feature_importance.csv"


# ============================================================
# Config
# ============================================================

TARGET_COL = "calibrated_regime_label_score"

STRICT_ALLOWED_FEATURES = [
    # unsafe lag history
    "total_unsafe_excess_lag1",
    "total_unsafe_excess_lag2",
    "total_unsafe_excess_rollmean_3",
    "total_unsafe_excess_rollmax_3",

    # blocked lag history
    "total_blocked_arrivals_lag1",
    "total_blocked_arrivals_lag2",
    "total_blocked_arrivals_rollmean_3",
    "total_blocked_arrivals_rollmax_3",

    # utilization lag history
    "max_utilization_ratio_lag1",
    "max_utilization_ratio_lag2",
    "max_utilization_ratio_rollmean_3",
    "max_utilization_ratio_rollmax_3",

    # unsafe rows history
    "num_unsafe_rows_lag1",
    "num_unsafe_rows_lag2",
    "num_unsafe_rows_rollmean_3",
    "num_unsafe_rows_rollmax_3",

    # overflow history
    "total_overflow_excess_lag1",
    "total_overflow_excess_lag2",
    "total_overflow_excess_rollmean_3",
    "total_overflow_excess_rollmax_3",

    # surge history
    "total_surge_gap_lag1",
    "total_surge_gap_lag2",
    "total_surge_gap_rollmean_3",
    "total_surge_gap_rollmax_3",

    # optional temporal context
    "time_index",
    "lookahead_horizon",
]

RANDOM_STATE = 42
TEST_SIZE = 0.25


# ============================================================
# Helpers
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ============================================================
# Dataset
# ============================================================

def build_dataset(df: pd.DataFrame):

    work = df.copy()

    if TARGET_COL not in work.columns:
        raise ValueError(f"{TARGET_COL} missing")

    y = work[TARGET_COL].astype(str)

    existing_features = [
        c for c in STRICT_ALLOWED_FEATURES
        if c in work.columns
    ]

    X = work[existing_features].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    numeric_cols = X.columns.tolist()
    categorical_cols = []

    return X, y, numeric_cols, categorical_cols


# ============================================================
# Pipeline
# ============================================================

def make_preprocessor(
    numeric_cols,
    categorical_cols,
):

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
        ]
    )


def make_models():

    return {

        "logistic_balanced": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# Feature importance
# ============================================================

def extract_feature_importance(
    model_pipeline,
    feature_names,
):

    try:

        model = model_pipeline.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        elif hasattr(model, "coef_"):
            importance = np.mean(
                np.abs(model.coef_),
                axis=0,
            )

        else:
            return pd.DataFrame()

        out = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        })

        out = out.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

        return out

    except Exception:
        return pd.DataFrame()


# ============================================================
# Main
# ============================================================

def main():

    print("\nCALIBRATED REGIME FORECAST (LAGGED STRICT)")
    print("=" * 70)

    df = safe_read_csv(DATA_PATH)

    X, y, numeric_cols, categorical_cols = build_dataset(df)

    print(f"\nRows used: {len(X)}")
    print(f"Target: {TARGET_COL}")
    print(f"Strict lagged features: {len(numeric_cols)}")

    print("\nFeatures used:")
    for c in numeric_cols:
        print(f" - {c}")

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = make_preprocessor(
        numeric_cols,
        categorical_cols,
    )

    models = make_models()

    results = []

    best_model_name = None
    best_f1 = -np.inf
    best_cm = None
    best_report = None
    best_importance = pd.DataFrame()

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)

        f1_macro = f1_score(
            y_test,
            pred,
            average="macro",
        )

        f1_weighted = f1_score(
            y_test,
            pred,
            average="weighted",
        )

        recall_macro = recall_score(
            y_test,
            pred,
            average="macro",
        )

        print(
            f"{model_name}: "
            f"accuracy={acc:.3f}, "
            f"f1_macro={f1_macro:.3f}, "
            f"f1_weighted={f1_weighted:.3f}, "
            f"recall_macro={recall_macro:.3f}"
        )

        model_path = (
            AI_MODELS_DIR
            / f"calibrated_regime_lagged__{model_name}.pkl"
        )

        save_pickle(pipe, model_path)

        results.append({
            "model_name": model_name,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "recall_macro": recall_macro,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": y.nunique(),
            "strict_lagged_only": True,
            "model_path": str(model_path),
        })

        if f1_macro > best_f1:

            best_f1 = f1_macro
            best_model_name = model_name

            best_cm = confusion_matrix(
                y_test,
                pred,
                labels=sorted(y.unique()),
            )

            best_report = classification_report(
                y_test,
                pred,
                digits=4,
            )

            best_importance = extract_feature_importance(
                pipe,
                numeric_cols,
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "f1_macro",
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(
        RESULTS_OUT,
        index=False,
    )

    # ============================================================
    # Confusion matrix
    # ============================================================

    if best_cm is not None:

        labels = sorted(y.unique())

        cm_df = pd.DataFrame(
            best_cm,
            index=[f"true_{x}" for x in labels],
            columns=[f"pred_{x}" for x in labels],
        )

        cm_df.to_csv(
            CONFUSION_OUT,
            index=True,
        )

    # ============================================================
    # Feature importance
    # ============================================================

    if not best_importance.empty:
        best_importance.to_csv(
            FEATURE_IMPORTANCE_OUT,
            index=False,
        )

    # ============================================================
    # Summary JSON
    # ============================================================

    summary = {
        "target_col": TARGET_COL,
        "best_model": best_model_name,
        "best_f1_macro": float(best_f1),
        "n_rows": int(len(df)),
        "n_classes": int(y.nunique()),
        "strict_lagged_only": True,
        "class_distribution": y.value_counts().to_dict(),
    }

    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # ============================================================
    # Report
    # ============================================================

    report_lines = []

    report_lines.append(
        "CALIBRATED REGIME FORECAST (STRICT LAGGED)\n"
    )

    report_lines.append(
        "=" * 70 + "\n"
    )

    report_lines.append(
        "Current operational metrics removed.\n"
    )

    report_lines.append(
        "Only lagged historical metrics retained.\n"
    )

    report_lines.append(
        f"Best model: {best_model_name}\n"
    )

    report_lines.append(
        f"Best macro F1: {best_f1:.4f}\n"
    )

    report_lines.append("\nClassification report:\n")

    if best_report is not None:
        report_lines.append(best_report)

    REPORT_OUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # ============================================================
    # Console
    # ============================================================

    print("\n=== Results ===")
    print(results_df.to_string(index=False))

    print("\n=== Best Model ===")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(RESULTS_OUT)
    print(CONFUSION_OUT)
    print(SUMMARY_OUT)
    print(REPORT_OUT)
    print(FEATURE_IMPORTANCE_OUT)


if __name__ == "__main__":
    main()