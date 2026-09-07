from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATA_PATH = Path("results/ai_datasets/forecast_dataset_combined.csv")
OUT_DIR = Path("results/ai_reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STRICT FEATURE SET (NO DESIGN / NO FUTURE / NO REGIME)
# ============================================================

STRICT_NUMERIC_FEATURES = [
    # current operational state
    "total_unsafe_excess",
    "total_blocked_arrivals",
    "max_utilization_ratio",
    "num_unsafe_rows",
    "total_overflow_excess",
    "total_surge_gap",

    # lag features
    "total_unsafe_excess_lag1",
    "total_unsafe_excess_lag2",
    "total_blocked_arrivals_lag1",
    "total_blocked_arrivals_lag2",
    "max_utilization_ratio_lag1",
    "max_utilization_ratio_lag2",

    # rolling
    "total_unsafe_excess_rollmean_3",
    "total_blocked_arrivals_rollmean_3",
    "max_utilization_ratio_rollmean_3",

    # flags
    "unsafe_high_flag",
    "blocked_high_flag",
    "utilization_critical_flag",
]


TARGET = "next_regime_label"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=[TARGET]).copy()

# Keep only safe columns
features = [c for c in STRICT_NUMERIC_FEATURES if c in df.columns]

X = df[features].copy()
y = df[TARGET].astype(str)

# Fill missing
X = X.fillna(0)


# ============================================================
# SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y if y.nunique() > 1 else None,
)


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
f1_macro = f1_score(y_test, preds, average="macro")
f1_weighted = f1_score(y_test, preds, average="weighted")


# ============================================================
# OUTPUT
# ============================================================

print("\nSTRICT REGIME MODEL")
print("=" * 60)
print(f"Accuracy: {acc:.3f}")
print(f"F1 Macro: {f1_macro:.3f}")
print(f"F1 Weighted: {f1_weighted:.3f}")

report = classification_report(y_test, preds)
print("\nClassification Report:\n", report)

(pd.DataFrame({
    "accuracy": [acc],
    "f1_macro": [f1_macro],
    "f1_weighted": [f1_weighted],
})).to_csv(OUT_DIR / "strict_regime_results.csv", index=False)

with open(OUT_DIR / "strict_regime_report.txt", "w") as f:
    f.write(report)

print("\nSaved → strict_regime_results.csv")