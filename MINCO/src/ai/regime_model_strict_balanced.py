from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("results/ai_datasets/forecast_dataset_combined.csv")
OUT_DIR = Path("results/ai_reports")
MODEL_DIR = Path("results/ai_models")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "next_regime_label"

RESULTS_PATH = OUT_DIR / "strict_balanced_regime_results.csv"
REPORT_PATH = OUT_DIR / "strict_balanced_regime_report.txt"
CONFUSION_PATH = OUT_DIR / "strict_balanced_regime_confusion_matrix.csv"
SUMMARY_PATH = OUT_DIR / "strict_balanced_regime_summary.json"


STRICT_NUMERIC_FEATURES = [
    "total_unsafe_excess",
    "total_blocked_arrivals",
    "max_utilization_ratio",
    "num_unsafe_rows",
    "total_overflow_excess",
    "total_surge_gap",
    "total_unsafe_excess_lag1",
    "total_unsafe_excess_lag2",
    "total_blocked_arrivals_lag1",
    "total_blocked_arrivals_lag2",
    "max_utilization_ratio_lag1",
    "max_utilization_ratio_lag2",
    "num_unsafe_rows_lag1",
    "num_unsafe_rows_lag2",
    "total_overflow_excess_lag1",
    "total_overflow_excess_lag2",
    "total_surge_gap_lag1",
    "total_surge_gap_lag2",
    "total_unsafe_excess_rollmean_3",
    "total_unsafe_excess_rollmax_3",
    "total_blocked_arrivals_rollmean_3",
    "total_blocked_arrivals_rollmax_3",
    "max_utilization_ratio_rollmean_3",
    "max_utilization_ratio_rollmax_3",
    "num_unsafe_rows_rollmean_3",
    "num_unsafe_rows_rollmax_3",
    "total_overflow_excess_rollmean_3",
    "total_overflow_excess_rollmax_3",
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


def add_strict_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in STRICT_NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    eps = 1e-6

    out["utilization_squared"] = out["max_utilization_ratio"] ** 2
    out["blocked_x_utilization"] = out["total_blocked_arrivals"] * out["max_utilization_ratio"]
    out["unsafe_x_utilization"] = out["total_unsafe_excess"] * out["max_utilization_ratio"]
    out["overflow_x_utilization"] = out["total_overflow_excess"] * out["max_utilization_ratio"]

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

    out["unsafe_momentum_1"] = out["total_unsafe_excess"] - out["total_unsafe_excess_lag1"]
    out["blocked_momentum_1"] = out["total_blocked_arrivals"] - out["total_blocked_arrivals_lag1"]
    out["utilization_momentum_1"] = out["max_utilization_ratio"] - out["max_utilization_ratio_lag1"]
    out["overflow_momentum_1"] = out["total_overflow_excess"] - out["total_overflow_excess_lag1"]
    out["surge_gap_momentum_1"] = out["total_surge_gap"] - out["total_surge_gap_lag1"]

    out["blocked_to_unsafe_ratio"] = out["total_blocked_arrivals"] / (
        out["total_unsafe_excess"].abs() + eps
    )

    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    engineered = [
        "utilization_squared",
        "blocked_x_utilization",
        "unsafe_x_utilization",
        "overflow_x_utilization",
        "unsafe_pressure_index",
        "access_pressure_index",
        "combined_stress_index",
        "unsafe_momentum_1",
        "blocked_momentum_1",
        "utilization_momentum_1",
        "overflow_momentum_1",
        "surge_gap_momentum_1",
        "blocked_to_unsafe_ratio",
    ]

    cols = [c for c in STRICT_NUMERIC_FEATURES + engineered if c in df.columns]

    forbidden = {
        TARGET,
        "current_regime",
        "current_regime_code",
        "scenario",
        "design_name",
        "dataset_source",
        "policy_name",
    }

    cols = [c for c in cols if c not in forbidden]
    cols = [c for c in cols if not c.startswith("target_next_")]
    cols = [c for c in cols if not c.startswith("next_")]

    return cols


def build_models() -> dict[str, object]:
    return {
        "logistic_balanced": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, str, pd.DataFrame]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    labels = sorted(y_test.unique().tolist())

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)
    recall_macro = recall_score(y_test, preds, average="macro", zero_division=0)

    report = classification_report(y_test, preds, zero_division=0)

    cm = confusion_matrix(y_test, preds, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    cm_df.insert(0, "model_name", model_name)

    row = {
        "model_name": model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(y_train.nunique()),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "recall_macro": float(recall_macro),
    }

    for label in labels:
        mask = y_test == label
        if mask.sum() > 0:
            row[f"support_{label}"] = int(mask.sum())

    model_path = MODEL_DIR / f"strict_balanced_regime__{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    row["model_path"] = str(model_path)

    return row, report, cm_df


def main() -> None:
    print("\nSTRICT BALANCED REGIME MODEL")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)

    if TARGET not in raw_df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df = add_strict_features(raw_df)
    df = df.dropna(subset=[TARGET]).copy()

    feature_cols = get_feature_cols(df)

    if not feature_cols:
        raise ValueError("No strict leakage-safe feature columns found.")

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[TARGET].astype(str)

    print(f"Rows used: {len(df)}")
    print(f"Features used: {len(feature_cols)}")
    print("\nTarget distribution:")
    print(y.value_counts().to_string())

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    rows = []
    reports = []
    cms = []

    for model_name, model in build_models().items():
        print(f"\nTraining {model_name}...", flush=True)

        row, report, cm_df = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        rows.append(row)
        reports.append(f"\nMODEL: {model_name}\n{'=' * 60}\n{report}")
        cms.append(cm_df)

        print(
            f"{model_name}: "
            f"accuracy={row['accuracy']:.3f}, "
            f"f1_macro={row['f1_macro']:.3f}, "
            f"f1_weighted={row['f1_weighted']:.3f}, "
            f"recall_macro={row['recall_macro']:.3f}",
            flush=True,
        )

    results_df = pd.DataFrame(rows).sort_values(
        "f1_macro",
        ascending=False,
    ).reset_index(drop=True)

    cm_all = pd.concat(cms, ignore_index=True) if cms else pd.DataFrame()

    best = results_df.iloc[0].to_dict() if not results_df.empty else {}

    results_df.to_csv(RESULTS_PATH, index=False)
    cm_all.to_csv(CONFUSION_PATH, index=False)
    REPORT_PATH.write_text("\n".join(reports), encoding="utf-8")

    summary = {
        "data_path": str(DATA_PATH),
        "target": TARGET,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "target_distribution": y.value_counts().to_dict(),
        "best_model": best,
        "results_path": str(RESULTS_PATH),
        "report_path": str(REPORT_PATH),
        "confusion_matrix_path": str(CONFUSION_PATH),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Results ===")
    print(results_df.to_string(index=False))

    print("\n=== Best model ===")
    print(json.dumps(best, indent=2))

    print("\nSaved:")
    print(RESULTS_PATH)
    print(REPORT_PATH)
    print(CONFUSION_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()