"""Deterministic multi-facility event generation for MINCO research mode."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config.paths import DATA_DIR
from src.hospital_events.models import EventType, HospitalEvent, SourceMode
from src.scenarios.operating_modes import get_operating_mode
from src.scenarios.public_reference import load_facility_reference


TIER_CAPACITY = {
    "tertiary": (48, 190, 10),
    "regional": (28, 110, 7),
    "community": (14, 68, 4),
}


def _event_id(mode: str, hospital_id: str, day: int, patient_index: int, kind: str) -> str:
    safe_hospital = hospital_id.replace("-", "_")
    return f"MINCO-SCALED-{mode}-{safe_hospital}-D{day:03d}-P{patient_index:05d}-{kind}"


def build_scaled_event_replay(
    *,
    n_hospitals: int = 120,
    horizon_days: int = 30,
    minimum_events: int = 100_000,
    operating_mode: str = "normal",
    seed: int = 20260906,
    start: datetime | None = None,
    public_reference_csv: str | Path | None = None,
) -> list[HospitalEvent]:
    """Generate a realistic-shaped, deterministic synthetic event lake.

    Each synthetic patient contributes arrival, bed-assignment and discharge
    lifecycle events. The generator continues until the requested minimum event
    count is met; default output is normally 100k+ events across 120 facilities.
    """
    if n_hospitals < 4 or horizon_days < 1 or minimum_events < 1:
        raise ValueError("scaled replay requires hospitals>=4, horizon_days>=1, minimum_events>=1")
    profile = get_operating_mode(operating_mode)
    if start is None:
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")

    catalog = load_facility_reference(
        n_hospitals=n_hospitals,
        csv_path=public_reference_csv,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    events: list[HospitalEvent] = []
    source_system = "minco_scaled_synthetic_engine"
    facility_provenance = catalog.provenance

    for row in catalog.facilities.to_dict("records"):
        hospital_id = str(row["hospital_id"])
        tier = str(row["hospital_type"])
        icu_base, ward_base, ed_servers = TIER_CAPACITY[tier]
        volume_factor = float(row["volume_factor"])
        # Capacity values are deliberately generated, not copied from CMS.
        for resource, capacity in (("ICU", icu_base), ("Ward", ward_base), ("ED", ed_servers)):
            events.append(HospitalEvent(
                event_id=f"MINCO-SCALED-{hospital_id}-CAP-{resource}",
                event_timestamp=start,
                ingested_at=start,
                hospital_id=hospital_id,
                event_type=EventType.CAPACITY_CHANGE,
                patient_pathway_or_cohort="system",
                resource=resource,
                quantity=float(capacity),
                source_system=source_system,
                source_mode=SourceMode.SYNTHETIC,
                metadata={
                    "operation": "set",
                    "operating_mode": profile.mode,
                    "hospital_tier": tier,
                    "capacity_claim": "synthetic_calibration",
                    "facility_provenance": facility_provenance["mode"],
                },
            ))

        for day in range(horizon_days):
            seasonal = 1.0 + 0.08 * np.sin((day / max(1, horizon_days - 1)) * 2.0 * np.pi)
            mean_arrivals = max(1.0, 10.0 * volume_factor * profile.demand_multiplier * seasonal)
            arrivals = int(rng.poisson(mean_arrivals))
            for patient_index in range(arrivals):
                minute = int(rng.integers(0, 24 * 60))
                arrival_time = start + timedelta(days=day, minutes=minute)
                age = float(np.clip(rng.normal(62.0, 17.0), 18.0, 95.0))
                acuity_score = float(np.clip(rng.beta(2.1, 2.8), 0.01, 0.99))
                shock_index_proxy = float(np.clip(rng.lognormal(mean=-0.15, sigma=0.36), 0.35, 2.5))
                oxygen_need_proxy = float(np.clip(rng.beta(1.7, 3.2), 0.0, 1.0))
                arrival_mode = str(rng.choice(["walk_in", "ambulance", "transfer"], p=[0.50, 0.38, 0.12]))
                escalation_logit = (
                    -3.2
                    + 2.35 * acuity_score
                    + 1.15 * oxygen_need_proxy
                    + 0.70 * (shock_index_proxy - 0.8)
                    + 0.012 * (age - 55.0)
                    + 0.45 * (arrival_mode == "ambulance")
                    + 0.32 * (arrival_mode == "transfer")
                    + 0.45 * np.log(profile.icu_multiplier)
                    + float(rng.normal(0.0, 0.55))
                )
                escalation_probability = 1.0 / (1.0 + np.exp(-escalation_logit))
                critical = bool(rng.random() < escalation_probability)
                resource = "ICU" if critical else "Ward"
                # Keep the pathway/cohort independent of the hidden escalation
                # outcome; using the outcome to name the cohort would leak the
                # label into the ML acceptance features.
                cohort = str(rng.choice(["medical", "respiratory", "cardiac", "surgical"], p=[0.38, 0.24, 0.22, 0.16]))
                patient_index_global = day * 10_000 + patient_index
                patient_id = f"SYN-{hospital_id}-{day:03d}-{patient_index:05d}"
                metadata = {
                    "synthetic": True,
                    "operating_mode": profile.mode,
                    "hospital_tier": tier,
                    "public_facility_id": row.get("public_facility_id"),
                    "facility_provenance": facility_provenance["mode"],
                    "calibration_basis": profile.calibration_basis,
                    "clinical_use": "not_validated_for_patient_care",
                    "age": age,
                    "acuity_score": acuity_score,
                    "shock_index_proxy": shock_index_proxy,
                    "oxygen_need_proxy": oxygen_need_proxy,
                    "arrival_mode": arrival_mode,
                    "icu_escalation_24h": int(critical),
                }
                events.append(HospitalEvent(
                    event_id=_event_id(profile.mode, hospital_id, day, patient_index_global, "ARR"),
                    event_timestamp=arrival_time,
                    ingested_at=start,
                    hospital_id=hospital_id,
                    event_type=EventType.ARRIVAL,
                    patient_pathway_or_cohort=cohort,
                    resource="ED",
                    quantity=1.0,
                    source_system=source_system,
                    source_mode=SourceMode.SYNTHETIC,
                    patient_id=patient_id,
                    metadata=metadata,
                ))
                assignment_time = arrival_time + timedelta(minutes=int(rng.integers(20, 150)))
                events.append(HospitalEvent(
                    event_id=_event_id(profile.mode, hospital_id, day, patient_index_global, "BED"),
                    event_timestamp=assignment_time,
                    ingested_at=start,
                    hospital_id=hospital_id,
                    event_type=EventType.BED_ASSIGNED,
                    patient_pathway_or_cohort=cohort,
                    resource=resource,
                    quantity=1.0,
                    source_system=source_system,
                    source_mode=SourceMode.SYNTHETIC,
                    patient_id=patient_id,
                    metadata=metadata,
                ))
                los_shape = 2.5 if resource == "Ward" else 2.8
                los_mean = (28.0 if resource == "Ward" else 44.0) * profile.length_of_stay_multiplier
                los_hours = max(2.0, float(rng.gamma(los_shape, los_mean / los_shape)))
                discharge_time = assignment_time + timedelta(hours=los_hours)
                events.append(HospitalEvent(
                    event_id=_event_id(profile.mode, hospital_id, day, patient_index_global, "DIS"),
                    event_timestamp=discharge_time,
                    ingested_at=start,
                    hospital_id=hospital_id,
                    event_type=EventType.DISCHARGE,
                    patient_pathway_or_cohort=cohort,
                    resource=resource,
                    quantity=1.0,
                    source_system=source_system,
                    source_mode=SourceMode.SYNTHETIC,
                    patient_id=patient_id,
                    metadata=metadata,
                ))

    events.sort(key=lambda event: (event.event_timestamp, event.event_id))
    if len(events) < minimum_events:
        raise RuntimeError(
            f"Scaled replay produced {len(events)} events, below requested minimum {minimum_events}; "
            "increase horizon_days or the facility catalog."
        )
    return events
