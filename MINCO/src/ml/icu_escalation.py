"""Calibrated ICU escalation classifier for synthetic operational validation.

The model estimates P(ICU escalation within 24h) from operational patient
features. It is intentionally separated from clinical diagnosis: the output is
a capacity-planning parameter for Monte Carlo scenario generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = (
    "age",
    "acuity_score",
    "shock_index_proxy",
    "oxygen_need_proxy",
    "arrival_mode",
    "cohort",
)
NUMERIC = ("age", "acuity_score", "shock_index_proxy", "oxygen_need_proxy")
CATEGORICAL = ("arrival_mode", "cohort")


@dataclass
class ICUEscalationModel:
    model: CalibratedClassifierCV | None = None

    @staticmethod
    def _base_pipeline() -> Pipeline:
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(NUMERIC)),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), list(CATEGORICAL)),
        ], sparse_threshold=0.0)
        return Pipeline([
            ("preprocess", pre),
            ("classifier", GradientBoostingClassifier(n_estimators=130, learning_rate=0.04, max_depth=2, random_state=20260817)),
        ])

    def fit(self, frame: pd.DataFrame) -> "ICUEscalationModel":
        missing = set(FEATURES + ("icu_escalation_24h",)) - set(frame.columns)
        if missing:
            raise ValueError(f"Escalation data missing columns: {sorted(missing)}")
        y = frame["icu_escalation_24h"].astype(int)
        if y.nunique() < 2:
            raise ValueError("Both escalation outcomes are required")
        self.model = CalibratedClassifierCV(self._base_pipeline(), method="sigmoid", cv=3)
        self.model.fit(frame[list(FEATURES)], y)
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        return self.model.predict_proba(frame[list(FEATURES)])[:, 1]


def synthetic_escalation_dataset(n_patients: int = 3000, *, seed: int = 20260817) -> pd.DataFrame:
    if n_patients < 200:
        raise ValueError("At least 200 patients are required")
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(61, 18, n_patients), 18, 96)
    acuity = np.clip(rng.beta(2.1, 2.8, n_patients), 0.01, 0.99)
    shock = np.clip(rng.lognormal(mean=-0.15, sigma=0.36, size=n_patients), 0.35, 2.5)
    oxygen = np.clip(rng.beta(1.7, 3.2, n_patients), 0.0, 1.0)
    arrival_mode = rng.choice(["walk_in", "ambulance", "transfer"], n_patients, p=[0.50, 0.38, 0.12])
    cohort = rng.choice(["medical", "respiratory", "cardiac", "surgical"], n_patients, p=[0.38, 0.24, 0.22, 0.16])
    logits = (
        -6.0
        + 4.6 * acuity
        + 2.4 * oxygen
        + 1.8 * (shock - 0.8)
        + 0.022 * (age - 55)
        + 1.15 * (arrival_mode == "ambulance")
        + 0.85 * (arrival_mode == "transfer")
        + 1.05 * (cohort == "respiratory")
        + 0.75 * (cohort == "cardiac")
        + 2.1 * acuity * oxygen
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    target = rng.binomial(1, probability)
    return pd.DataFrame({
        "age": age,
        "acuity_score": acuity,
        "shock_index_proxy": shock,
        "oxygen_need_proxy": oxygen,
        "arrival_mode": arrival_mode,
        "cohort": cohort,
        "icu_escalation_24h": target,
    })


def evaluate_escalation_model(model: ICUEscalationModel, frame: pd.DataFrame) -> dict[str, float]:
    y = frame["icu_escalation_24h"].astype(int).to_numpy()
    p = model.predict_probability(frame)
    constant = np.full_like(p, y.mean(), dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "constant_brier_score": float(brier_score_loss(y, constant)),
        "prevalence": float(y.mean()),
    }
