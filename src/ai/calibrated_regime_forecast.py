from __future__ import annotations

from pathlib import Path
import json
import pickle
import time

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

AI_DATA_DIR = RESULTS_DIR / "ai_datasets"
AI_REPORTS_DIR = RESULTS_DIR / "ai_reports"
AI_MODELS_DIR = RESULTS_DIR / "ai_models"
FIGURES_DIR = RESULTS_DIR / "figures"

AI_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = RESULTS_DIR / "tables" / "regime_recalibration_row_level.csv"

RESULTS_OUT = AI_REPORTS_DIR / "calibrated_regime_forecast_results.csv"
CONFUSION_OUT = AI_REPORTS_DIR / "calibrated_regime_confusion_matrix.csv"
SUMMARY_OUT = AI_REPORTS_DIR / "calibrated_regime_forecast_summary.json"
REPORT_OUT = AI_REPORTS_DIR / "calibrated_regime_forecast_report.txt"
FEATURE_IMPORTANCE_OUT = AI_REPORTS_DIR / "calibrated_regime_feature_importance.csv"


# ============================================================
# Config
# ============================================================

TARGET_COL = "calibrated_regime_label_score"

DROP_COLS = [
    "next_regime_label",
    "original_next_regime_label",
    "calibrated_regime_label_rule",
    "calibrated_regime_label_score",
    "label_changed_rule",
    "label_changed_score",
    "operational_regime_score",  # this is a direct, non-overlapping threshold-bucketing
                                  # source of calibrated_regime_label_score itself (verified:
                                  # normal=0-0.75, surge=0.76-1.99, crisis=2.0+, zero overlap)
                                  # -- leaving it in let the model "predict" by just re-deriving
                                  # its own label, producing meaningless 97-100% accuracy.
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


def safe_float(x, default=np.nan):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


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

    drop_existing = [c for c in DROP_COLS if c in work.columns]
    X = work.drop(columns=drop_existing)

    leakage_cols = [
        c for c in X.columns
        if "future" in c.lower()
        or "target" in c.lower()
        or "next_" in c.lower()
    ]

    if leakage_cols:
        X = X.drop(columns=leakage_cols)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols
    ]

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

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
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
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,  # was -1; on Windows, joblib's process-spawn parallelism on a
                       # subprocess-launched script is a known catastrophic-slowdown
                       # pattern -- dataset is only ~11k rows, single-threaded is plenty
        ),

        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,       # was None -- unbounded depth + bootstrap=False (ExtraTrees'
                                 # default) means every tree trains on the full dataset with
                                 # randomized splits and no depth cap, which is the likely
                                 # cause of the runaway runtime. Capping depth bounds worst-case
                                 # tree size without materially hurting accuracy at this data size.
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
            bootstrap=True,      # was unset (defaults to False for ExtraTrees, unlike
                                  # RandomForest) -- matching RandomForest's bootstrapping
                                  # behavior so each tree sees a resampled subset, not the
                                  # full 10,900 rows every time.
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
    numeric_cols,
    categorical_cols,
):
    try:
        preprocess = model_pipeline.named_steps["preprocess"]
        model = model_pipeline.named_steps["model"]

        feature_names = []

        feature_names.extend(numeric_cols)

        if categorical_cols:
            encoder = (
                preprocess.named_transformers_["cat"]
                .named_steps["encoder"]
            )

            encoded = encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(encoded.tolist())

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

    print("\nCALIBRATED REGIME FORECAST")
    print("=" * 70)

    df = safe_read_csv(DATA_PATH)

    X, y, numeric_cols, categorical_cols = build_dataset(df)

    print(f"\nRows used: {len(X)}")
    print(f"Target: {TARGET_COL}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

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
    best_pipeline = None
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

        _fit_start = time.perf_counter()
        pipe.fit(X_train, y_train)
        _fit_seconds = time.perf_counter() - _fit_start
        print(f"  {model_name} fit in {_fit_seconds:.2f}s")

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
            / f"calibrated_regime_forecast__{model_name}.pkl"
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
            "model_path": str(model_path),
        })

        if f1_macro > best_f1:

            best_f1 = f1_macro
            best_model_name = model_name
            best_pipeline = pipe

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
                categorical_cols,
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
        "class_distribution": y.value_counts().to_dict(),
    }

    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # ============================================================
    # Report text
    # ============================================================

    report_lines = []

    report_lines.append(
        "CALIBRATED REGIME FORECAST REPORT\n"
    )

    report_lines.append(
        "=" * 70 + "\n"
    )

    report_lines.append(
        f"Target column: {TARGET_COL}\n"
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