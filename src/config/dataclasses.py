from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STATE_ORDER: list[str] = ["ED", "ICU", "Ward", "Discharged", "Dead"]


@dataclass(frozen=True)
class HospitalsData:
    df: pd.DataFrame


@dataclass(frozen=True)
class CapacitiesData:
    df: pd.DataFrame


@dataclass(frozen=True)
class SurgeCapsData:
    df: pd.DataFrame


@dataclass(frozen=True)
class SafeThresholdsData:
    df: pd.DataFrame


@dataclass(frozen=True)
class TransferLanesData:
    df: pd.DataFrame


@dataclass(frozen=True)
class CostsData:
    df: pd.DataFrame


@dataclass(frozen=True)
class ElectiveBoundsData:
    df: pd.DataFrame


@dataclass(frozen=True)
class ArrivalsData:
    df: pd.DataFrame


@dataclass(frozen=True)
class TransitionMatrixData:
    cohort: str
    df: pd.DataFrame


@dataclass(frozen=True)
class TransitionBundle:
    matrices: dict[str, TransitionMatrixData]


@dataclass(frozen=True)
class HealthcareInstance:
    hospitals: HospitalsData
    capacities: CapacitiesData
    surge_caps: SurgeCapsData
    safe_thresholds: SafeThresholdsData
    transfer_lanes: TransferLanesData
    costs: CostsData
    elective_bounds: ElectiveBoundsData
    arrivals: ArrivalsData
    transitions: TransitionBundle
