from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

DATA_PATH = AI_DATA_DIR / "forecast_dataset_combined.csv"

TARGET_UNSAFE_COL = "target_next_total_unsafe_excess"

RESULTS_PATH = AI_REPORTS_DIR / "unsafe_excess_risk_classifier_results.csv"
SUMMARY_PATH = AI_REPORTS_DIR / "unsafe_excess_risk_classifier_summary.json"
REPORT_PATH = AI_REPORTS_DIR / "unsafe_excess_risk_classifier_reports.txt"
CONFUSION_PATH = AI_REPORTS_DIR / "unsafe_excess_risk_classifier_confusion_matrices.csv"
FEATURE_IMPORTANCE_PATH = AI_REPORTS_DIR / "unsafe_excess_risk_classifier_feature_importance.csv"

FIG_TARGET_DIST_PATH = FIGURES_DIR / "unsafe_excess_risk_target_distribution.png"
FIG_FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "unsafe_excess_risk_feature_importance.png"


# ============================================================
# Leakage-safe feature configuration
# ============================================================

FORBIDDEN_EXACT_COLUMNS = {
    TARGET_UNSAFE_COL,
    "target_next_total_blocked_arrivals",
    "target_next_max_utilization_ratio",
    "target_next_num_unsafe_rows",
    "target_next_total_overflow_excess",
    "target_next_total_surge_gap",
    "target_next_utilization_critical_flag",
    "target_next_unsafe_positive_flag",
    "next_regime_label",
    "current_regime",
    "current_regime_code",
}

FORBIDDEN_PREFIXES = [
    "target_next_",
    "next_",
]

NUMERIC_BASE_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "policy_name",
    "scenario",
    "design_name",
    "dataset_source",
]


# ============================================================
# Targets
# ============================================================

RISK_TARGETS = {
    "unsafe_gt_5": {
        "target_col": "target_next_unsafe_gt_5",
        "threshold": 5.0,
        "description": "Next unsafe excess greater than 5",
    },
    "unsafe_gt_10": {
        "target_col": "target_next_unsafe_gt_10",
        "threshold": 10.0,
        "description": "Next unsafe excess greater than 10",
    },
    "unsafe_top_quartile": {
        "target_col": "target_next_unsafe_top_quartile",
        "threshold": None,
        "description": "Next unsafe excess in the empirical top quartile",
    },
}


# ============================================================
# IO helpers
# ============================================================

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ============================================================
# Feature engineering
# ============================================================

def is_forbidden_feature(col: str) -> bool:
    if col in FORBIDDEN_EXACT_COLUMNS:
        return True

    for prefix in FORBIDDEN_PREFIXES:
        if col.startswith(prefix):
            return True

    return False


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in NUMERIC_BASE_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].astype(str).fillna("unknown")

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

    out["stress_acceleration_index"] = (
        out["unsafe_momentum_1"]
        + out["overflow_momentum_1"]
        + 2.0 * out["utilization_momentum_1"]
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_risk_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if TARGET_UNSAFE_COL not in out.columns:
        raise ValueError(f"Missing target column: {TARGET_UNSAFE_COL}")

    out[TARGET_UNSAFE_COL] = pd.to_numeric(out[TARGET_UNSAFE_COL], errors="coerce")

    out["target_next_unsafe_gt_5"] = (out[TARGET_UNSAFE_COL] > 5.0).astype(int)
    out["target_next_unsafe_gt_10"] = (out[TARGET_UNSAFE_COL] > 10.0).astype(int)

    q75 = float(out[TARGET_UNSAFE_COL].dropna().quantile(0.75))
    RISK_TARGETS["unsafe_top_quartile"]["threshold"] = q75
    out["target_next_unsafe_top_quartile"] = (out[TARGET_UNSAFE_COL] >= q75).astype(int)

    return out


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
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
        "stress_acceleration_index",
    ]

    numeric_features = [
        c
        for c in NUMERIC_BASE_FEATURES + engineered_numeric
        if c in df.columns and not is_forbidden_feature(c)
    ]

    categorical_features = [
        c
        for c in CATEGORICAL_FEATURES
        if c in df.columns and not is_forbidden_feature(c)
    ]

    return numeric_features, categorical_features


# ============================================================
# Models
# ============================================================

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
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
        "logistic_balanced": Pipeline(
            steps=[
                ("preprocess", linear_pre),
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
        "random_forest_balanced": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=None,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=450,
                        max_depth=None,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocess", tree_pre),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=250,
                        learning_rate=0.04,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# Evaluation
# ============================================================

def classification_metrics(y_true: np.ndarray, preds: np.ndarray, prob_pos: np.ndarray | None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "precision_pos": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
    }

    if prob_pos is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            out["roc_auc"] = np.nan
    else:
        out["roc_auc"] = np.nan

    return out


def get_positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    try:
        probs = model.predict_proba(X)
        classes = model.named_steps["model"].classes_
        class_list = list(classes)

        if 1 in class_list:
            idx = class_list.index(1)
        elif "1" in class_list:
            idx = class_list.index("1")
        else:
            return None

        return probs[:, idx]
    except Exception:
        return None


def confusion_df(model_name: str, target_name: str, y_true: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    return pd.DataFrame(
        [
            {
                "target_name": target_name,
                "model_name": model_name,
                "true_negative": int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive": int(cm[1, 1]),
            }
        ]
    )


def extract_feature_names(
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    names = []
    names.extend(numeric_features)

    try:
        pre = model.named_steps["preprocess"]
        onehot = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_names = onehot.get_feature_names_out(categorical_features).tolist()
        names.extend(cat_names)
    except Exception:
        names.extend(categorical_features)

    return names


def extract_feature_importance(
    target_name: str,
    model_name: str,
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    feature_names = extract_feature_names(
        model=model,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    importances = estimator.feature_importances_
    n = min(len(feature_names), len(importances))

    out = pd.DataFrame(
        {
            "target_name": target_name,
            "model_name": model_name,
            "feature": feature_names[:n],
            "importance": importances[:n],
        }
    )

    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# ============================================================
# Plots
# ============================================================

def plot_target_distribution(df: pd.DataFrame) -> None:
    rows = []

    for target_name, spec in RISK_TARGETS.items():
        target_col = spec["target_col"]
        if target_col not in df.columns:
            continue

        y = pd.to_numeric(df[target_col], errors="coerce").dropna()
        rows.append(
            {
                "target": target_name,
                "positive_share": float((y == 1).mean()),
                "negative_share": float((y == 0).mean()),
            }
        )

    if not rows:
        return

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(plot_df)), plot_df["positive_share"])
    plt.xticks(range(len(plot_df)), plot_df["target"], rotation=20, ha="right")
    plt.ylabel("Positive class share")
    plt.title("Unsafe Excess Risk Target Positive-Rate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_TARGET_DIST_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    if importance_df.empty:
        return

    best_target = importance_df["target_name"].iloc[0]
    best_model = importance_df["model_name"].iloc[0]

    top = (
        importance_df[
            (importance_df["target_name"] == best_target)
            & (importance_df["model_name"] == best_model)
        ]
        .head(20)
        .copy()
        .sort_values("importance", ascending=True)
    )

    if top.empty:
        return

    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Unsafe Risk Feature Importance\n{best_target} / {best_model}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_FEATURE_IMPORTANCE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Training
# ============================================================

def train_one_target(
    df: pd.DataFrame,
    target_name: str,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    work = df.dropna(subset=[target_col]).copy()
    y = work[target_col].astype(int)
    X = work[numeric_features + categorical_features].copy()

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    models = build_models(numeric_features, categorical_features)

    result_rows = []
    confusion_frames = []
    importance_frames = []
    report_blocks = []

    for model_name, model in models.items():
        print(f"\nTraining {target_name} / {model_name}...", flush=True)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        prob_pos = get_positive_probability(model, X_test)

        metrics = classification_metrics(y_test.to_numpy(), preds, prob_pos)

        model_path = AI_MODELS_DIR / f"unsafe_risk__{target_name}__{model_name}.pkl"
        save_pickle(model, model_path)

        pos_share = float(y.mean())

        result_rows.append(
            {
                "target_name": target_name,
                "target_col": target_col,
                "threshold": RISK_TARGETS[target_name]["threshold"],
                "positive_share": pos_share,
                "model_name": model_name,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "n_numeric_features": int(len(numeric_features)),
                "n_categorical_features": int(len(categorical_features)),
                **metrics,
                "model_path": str(model_path),
                "leakage_safe": True,
            }
        )

        confusion_frames.append(
            confusion_df(
                model_name=model_name,
                target_name=target_name,
                y_true=y_test.to_numpy(),
                preds=preds,
            )
        )

        imp = extract_feature_importance(
            target_name=target_name,
            model_name=model_name,
            model=model,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        if not imp.empty:
            importance_frames.append(imp)

        report_blocks.append(
            "\n".join(
                [
                    f"TARGET: {target_name}",
                    f"MODEL: {model_name}",
                    "=" * 70,
                    classification_report(y_test, preds, zero_division=0),
                ]
            )
        )

        print(
            f"{model_name}: "
            f"acc={metrics['accuracy']:.3f}, "
            f"f1_pos={metrics['f1_pos']:.3f}, "
            f"recall_pos={metrics['recall_pos']:.3f}, "
            f"f1_macro={metrics['f1_macro']:.3f}, "
            f"auc={metrics['roc_auc']:.3f}",
            flush=True,
        )

    result_df = pd.DataFrame(result_rows)
    confusion_all = pd.concat(confusion_frames, ignore_index=True) if confusion_frames else pd.DataFrame()
    importance_all = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()

    return result_df, confusion_all, importance_all, report_blocks


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nUNSAFE EXCESS RISK CLASSIFIER")
    print("=" * 70)

    raw_df = safe_read_csv(DATA_PATH)
    df = add_engineered_features(raw_df)
    df = add_risk_targets(df)

    numeric_features, categorical_features = get_feature_lists(df)

    print(f"Rows: {len(df)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    all_results = []
    all_confusions = []
    all_importance = []
    all_reports = []

    for target_name, spec in RISK_TARGETS.items():
        target_col = spec["target_col"]

        result_df, confusion_df_all, importance_df, report_blocks = train_one_target(
            df=df,
            target_name=target_name,
            target_col=target_col,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        all_results.append(result_df)
        all_confusions.append(confusion_df_all)
        if not importance_df.empty:
            all_importance.append(importance_df)
        all_reports.extend(report_blocks)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    confusion_all = pd.concat(all_confusions, ignore_index=True) if all_confusions else pd.DataFrame()
    importance_all = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()

    results_df = results_df.sort_values(
        ["target_name", "f1_pos", "roc_auc"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_PATH, index=False)
    confusion_all.to_csv(CONFUSION_PATH, index=False)
    importance_all.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    REPORT_PATH.write_text("\n\n".join(all_reports), encoding="utf-8")

    plot_target_distribution(df)
    plot_feature_importance(importance_all)

    best_by_target = []
    if not results_df.empty:
        for target_name in results_df["target_name"].dropna().unique():
            sub = results_df[results_df["target_name"] == target_name].copy()
            best = sub.sort_values(["f1_pos", "roc_auc"], ascending=[False, False]).iloc[0]
            best_by_target.append(
                {
                    "target_name": target_name,
                    "best_model": best["model_name"],
                    "threshold": best["threshold"],
                    "positive_share": best["positive_share"],
                    "f1_pos": best["f1_pos"],
                    "recall_pos": best["recall_pos"],
                    "precision_pos": best["precision_pos"],
                    "f1_macro": best["f1_macro"],
                    "roc_auc": best["roc_auc"],
                }
            )

    summary = {
        "input_path": str(DATA_PATH),
        "target_source_col": TARGET_UNSAFE_COL,
        "n_rows": int(len(df)),
        "n_numeric_features": int(len(numeric_features)),
        "n_categorical_features": int(len(categorical_features)),
        "targets": RISK_TARGETS,
        "best_by_target": best_by_target,
        "outputs": {
            "results": str(RESULTS_PATH),
            "summary": str(SUMMARY_PATH),
            "reports": str(REPORT_PATH),
            "confusion_matrices": str(CONFUSION_PATH),
            "feature_importance": str(FEATURE_IMPORTANCE_PATH),
            "target_distribution_figure": str(FIG_TARGET_DIST_PATH),
            "feature_importance_figure": str(FIG_FEATURE_IMPORTANCE_PATH),
        },
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Best by target ===")
    print(pd.DataFrame(best_by_target).to_string(index=False))

    print("\n=== Full results ===")
    print(results_df.to_string(index=False))

    print("\nSaved:")
    print(RESULTS_PATH)
    print(SUMMARY_PATH)
    print(REPORT_PATH)
    print(CONFUSION_PATH)
    print(FEATURE_IMPORTANCE_PATH)
    print(FIG_TARGET_DIST_PATH)
    print(FIG_FEATURE_IMPORTANCE_PATH)


if __name__ == "__main__":
    main()