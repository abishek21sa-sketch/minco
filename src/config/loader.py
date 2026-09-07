from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from src.config.dataclasses import (
    ArrivalsData,
    CapacitiesData,
    CostsData,
    ElectiveBoundsData,
    HealthcareInstance,
    HospitalsData,
    SafeThresholdsData,
    SurgeCapsData,
    TransferLanesData,
    TransitionBundle,
    TransitionMatrixData,
)

BASE_INSTANCE_FILES = {
    "hospitals": "hospitals.csv",
    "capacities": "capacities.csv",
    "surge_caps": "surge_caps.csv",
    "safe_thresholds": "safe_thresholds.csv",
    "transfer_lanes": "transfer_lanes.csv",
    "costs": "costs.csv",
    "elective_bounds": "elective_bounds.csv",
    "arrivals": "arrivals.csv",
}

TRANSITION_FILES = {
    "c1": "cohort_c1.csv",
    "c2": "cohort_c2.csv",
    "c3": "cohort_c3.csv",
    "c4": "cohort_c4.csv",
}

REQUIRED_COLUMNS = {
    "hospitals": ["hospital_id", "hospital_name", "hospital_type"],
    "capacities": ["hospital_id", "resource", "base_capacity"],
    "surge_caps": ["hospital_id", "resource", "max_surge"],
    "safe_thresholds": ["hospital_id", "resource", "safe_utilization"],
    "transfer_lanes": [
        "from_hospital",
        "to_hospital",
        "allowed",
        "transfer_capacity",
        "transfer_cost",
        "transfer_time",
    ],
    "costs": ["cost_name", "value"],
    "elective_bounds": ["day", "hospital_id", "cohort", "min_elective", "max_elective"],
    "arrivals": ["day", "hospital_id", "cohort", "arrivals"],
    "transitions": ["from_state", "to_state", "prob"],
}

UNIQUE_KEYS: dict[str, Sequence[str]] = {
    "hospitals": ["hospital_id"],
    "capacities": ["hospital_id", "resource"],
    "surge_caps": ["hospital_id", "resource"],
    "safe_thresholds": ["hospital_id", "resource"],
    "transfer_lanes": ["from_hospital", "to_hospital"],
    "costs": ["cost_name"],
    "elective_bounds": ["day", "hospital_id", "cohort"],
    "arrivals": ["day", "hospital_id", "cohort"],
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _validate_required_columns(
    df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _validate_no_missing_values(
    df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str
) -> None:
    null_cols = [col for col in required_columns if df[col].isnull().any()]
    if null_cols:
        raise ValueError(f"{dataset_name} contains missing values in required columns: {null_cols}")


def _validate_nonnegative(
    df: pd.DataFrame, numeric_columns: Iterable[str], dataset_name: str
) -> None:
    bad_cols: list[str] = [col for col in numeric_columns if (df[col] < 0).any()]
    if bad_cols:
        raise ValueError(f"{dataset_name} contains negative values in columns: {bad_cols}")


def _validate_probability_bounds(df: pd.DataFrame, prob_col: str, dataset_name: str) -> None:
    if ((df[prob_col] < 0) | (df[prob_col] > 1)).any():
        raise ValueError(
            f"{dataset_name} has probability values outside [0, 1] in column '{prob_col}'"
        )


def _validate_unique_keys(df: pd.DataFrame, key_columns: Sequence[str], dataset_name: str) -> None:
    duplicates = df[df.duplicated(list(key_columns), keep=False)]
    if not duplicates.empty:
        sample = duplicates[list(key_columns)].drop_duplicates().head(10).to_dict("records")
        raise ValueError(
            f"{dataset_name} contains duplicate business keys {list(key_columns)}. "
            f"Examples: {sample}"
        )


def _validate_base_table_rules(dataset_name: str, df: pd.DataFrame) -> None:
    if dataset_name == "capacities":
        _validate_nonnegative(df, ["base_capacity"], dataset_name)
    elif dataset_name == "surge_caps":
        _validate_nonnegative(df, ["max_surge"], dataset_name)
    elif dataset_name == "safe_thresholds":
        _validate_probability_bounds(df, "safe_utilization", dataset_name)
        if (df["safe_utilization"] <= 0).any():
            raise ValueError("safe_thresholds.safe_utilization must be greater than zero")
    elif dataset_name == "transfer_lanes":
        _validate_nonnegative(
            df, ["transfer_capacity", "transfer_cost", "transfer_time"], dataset_name
        )
        invalid_allowed = sorted(set(df["allowed"].tolist()) - {0, 1})
        if invalid_allowed:
            raise ValueError(f"transfer_lanes.allowed must be binary; found {invalid_allowed}")
        if (df["from_hospital"] == df["to_hospital"]).any():
            raise ValueError("transfer_lanes cannot contain self-transfer arcs")
    elif dataset_name == "costs":
        _validate_nonnegative(df, ["value"], dataset_name)
    elif dataset_name == "elective_bounds":
        _validate_nonnegative(df, ["day", "min_elective", "max_elective"], dataset_name)
        if (df["min_elective"] > df["max_elective"]).any():
            raise ValueError("elective_bounds contains min_elective > max_elective")
    elif dataset_name == "arrivals":
        _validate_nonnegative(df, ["day", "arrivals"], dataset_name)


def _load_base_instance_tables(base_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for dataset_name, filename in BASE_INSTANCE_FILES.items():
        df = _read_csv(base_dir / filename)
        _validate_required_columns(df, REQUIRED_COLUMNS[dataset_name], dataset_name)
        _validate_no_missing_values(df, REQUIRED_COLUMNS[dataset_name], dataset_name)
        _validate_unique_keys(df, UNIQUE_KEYS[dataset_name], dataset_name)
        _validate_base_table_rules(dataset_name, df)
        tables[dataset_name] = df
    _validate_referential_integrity(tables)
    return tables


def _validate_referential_integrity(tables: dict[str, pd.DataFrame]) -> None:
    hospitals = set(tables["hospitals"]["hospital_id"].astype(str))

    def check(dataset: str, column: str) -> None:
        unknown = sorted(set(tables[dataset][column].astype(str)) - hospitals)
        if unknown:
            raise ValueError(f"{dataset}.{column} references unknown hospitals: {unknown}")

    for dataset in ["capacities", "surge_caps", "safe_thresholds", "elective_bounds", "arrivals"]:
        check(dataset, "hospital_id")
    check("transfer_lanes", "from_hospital")
    check("transfer_lanes", "to_hospital")

    capacity_keys = set(
        map(tuple, tables["capacities"][["hospital_id", "resource"]].astype(str).to_numpy())
    )
    for dataset in ["surge_caps", "safe_thresholds"]:
        keys = set(map(tuple, tables[dataset][["hospital_id", "resource"]].astype(str).to_numpy()))
        if keys != capacity_keys:
            missing = sorted(capacity_keys - keys)
            extra = sorted(keys - capacity_keys)
            raise ValueError(
                f"{dataset} resource keys do not match capacities; missing={missing}, extra={extra}"
            )


def _load_transition_tables(transitions_dir: Path) -> TransitionBundle:
    from src.markov.transition_utils import validate_transition_dataframe

    matrices: dict[str, TransitionMatrixData] = {}
    for cohort, filename in TRANSITION_FILES.items():
        df = _read_csv(transitions_dir / filename)
        _validate_required_columns(df, REQUIRED_COLUMNS["transitions"], f"transitions[{cohort}]")
        _validate_no_missing_values(df, REQUIRED_COLUMNS["transitions"], f"transitions[{cohort}]")
        validate_transition_dataframe(df=df, cohort=cohort)
        matrices[cohort] = TransitionMatrixData(cohort=cohort, df=df)
    return TransitionBundle(matrices=matrices)


def load_healthcare_instance(data_dir: str | Path = "data") -> HealthcareInstance:
    """Load and validate one MINCO healthcare instance."""
    data_path = Path(data_dir)
    base_tables = _load_base_instance_tables(data_path / "base_instance")
    transition_bundle = _load_transition_tables(data_path / "transitions")
    return HealthcareInstance(
        hospitals=HospitalsData(df=base_tables["hospitals"]),
        capacities=CapacitiesData(df=base_tables["capacities"]),
        surge_caps=SurgeCapsData(df=base_tables["surge_caps"]),
        safe_thresholds=SafeThresholdsData(df=base_tables["safe_thresholds"]),
        transfer_lanes=TransferLanesData(df=base_tables["transfer_lanes"]),
        costs=CostsData(df=base_tables["costs"]),
        elective_bounds=ElectiveBoundsData(df=base_tables["elective_bounds"]),
        arrivals=ArrivalsData(df=base_tables["arrivals"]),
        transitions=transition_bundle,
    )
