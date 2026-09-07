"""Governed operating-mode catalog for the scaled MINCO research twin.

The modes are stress-test parameterizations, not clinical disease classifiers.
They make the provenance boundary explicit: public surveillance can calibrate
direction and magnitude, while facility capacity, patient pathways and outcomes
remain synthetic until a real deployment validation program exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


OperatingMode = Literal[
    "normal",
    "lower_demand",
    "respiratory_surge",
    "covid_like",
    "flu_like",
    "mixed_surge",
]


@dataclass(frozen=True)
class OperatingModeProfile:
    mode: OperatingMode
    display_name: str
    description: str
    demand_multiplier: float
    ed_multiplier: float
    icu_multiplier: float
    length_of_stay_multiplier: float
    transfer_multiplier: float
    calibration_basis: str
    evidence_boundary: str = (
        "Synthetic research parameterization; not a diagnosis, forecast of an individual, "
        "or authorization for patient-care operations."
    )


MODE_CATALOG: dict[str, OperatingModeProfile] = {
    "normal": OperatingModeProfile(
        mode="normal",
        display_name="Normal operations",
        description="Reference seasonal operating pressure around the synthetic baseline.",
        demand_multiplier=1.00,
        ed_multiplier=1.00,
        icu_multiplier=1.00,
        length_of_stay_multiplier=1.00,
        transfer_multiplier=1.00,
        calibration_basis="Synthetic baseline anchored to the bundled Meridian reference case.",
    ),
    "lower_demand": OperatingModeProfile(
        mode="lower_demand",
        display_name="Lower demand",
        description="Lower-arrival period with reduced ED and critical-care pressure.",
        demand_multiplier=0.78,
        ed_multiplier=0.82,
        icu_multiplier=0.86,
        length_of_stay_multiplier=0.96,
        transfer_multiplier=1.05,
        calibration_basis="Synthetic low-regime stress test; calibrate against local historical seasonality.",
    ),
    "respiratory_surge": OperatingModeProfile(
        mode="respiratory_surge",
        display_name="Respiratory surge",
        description="Network-wide respiratory pressure with elevated ED throughput and ICU demand.",
        demand_multiplier=1.35,
        ed_multiplier=1.42,
        icu_multiplier=1.30,
        length_of_stay_multiplier=1.10,
        transfer_multiplier=0.92,
        calibration_basis="Directionally calibrated to CDC RESP-NET/FluView aggregate surveillance.",
    ),
    "covid_like": OperatingModeProfile(
        mode="covid_like",
        display_name="COVID-like stress",
        description="Respiratory stress test with longer stays, higher ICU mix and constrained transfers.",
        demand_multiplier=1.55,
        ed_multiplier=1.60,
        icu_multiplier=1.65,
        length_of_stay_multiplier=1.35,
        transfer_multiplier=0.78,
        calibration_basis="Synthetic COVID-like scenario calibrated only from public aggregate respiratory trends.",
    ),
    "flu_like": OperatingModeProfile(
        mode="flu_like",
        display_name="Flu-like seasonal peak",
        description="Seasonal influenza-like pressure with higher arrival volume and moderate ICU lift.",
        demand_multiplier=1.28,
        ed_multiplier=1.34,
        icu_multiplier=1.18,
        length_of_stay_multiplier=1.06,
        transfer_multiplier=0.95,
        calibration_basis="Directionally calibrated to CDC FluView/FluSurv-NET aggregate surveillance.",
    ),
    "mixed_surge": OperatingModeProfile(
        mode="mixed_surge",
        display_name="Mixed respiratory surge",
        description="Combined stress case for respiratory, seasonal and transfer-network pressure.",
        demand_multiplier=1.72,
        ed_multiplier=1.78,
        icu_multiplier=1.58,
        length_of_stay_multiplier=1.24,
        transfer_multiplier=0.70,
        calibration_basis="Synthetic composite stress test; requires local validation before use.",
    ),
}


def get_operating_mode(mode: str | None) -> OperatingModeProfile:
    key = (mode or "normal").strip().lower()
    try:
        return MODE_CATALOG[key]
    except KeyError as exc:
        allowed = ", ".join(MODE_CATALOG)
        raise ValueError(f"Unknown operating_mode={mode!r}; expected one of: {allowed}") from exc


def operating_mode_catalog() -> list[dict[str, object]]:
    """Return a JSON-safe catalog suitable for API and native-client display."""
    return [asdict(profile) for profile in MODE_CATALOG.values()]
