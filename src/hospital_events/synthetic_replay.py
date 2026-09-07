"""Generate a deterministic canonical event replay from the bundled reference case.

This is explicitly synthetic. It exists so the full state/replay product can be
validated without pretending that the repository contains hospital ADT data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config.loader import load_healthcare_instance
from src.config.paths import DATA_DIR
from src.hospital_events.models import EventType, HospitalEvent, SourceMode


def build_reference_event_replay(
    *,
    data_dir: Path = DATA_DIR,
    start: datetime | None = None,
) -> list[HospitalEvent]:
    instance = load_healthcare_instance(data_dir)
    if start is None:
        start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")

    events: list[HospitalEvent] = []
    # Seed explicit capacities before arrivals so state reconstruction is causal.
    for row in instance.capacities.df.to_dict("records"):
        events.append(HospitalEvent(
            event_timestamp=start,
            hospital_id=str(row["hospital_id"]),
            event_type=EventType.CAPACITY_CHANGE,
            patient_pathway_or_cohort="system",
            resource=str(row["resource"]),
            quantity=float(row["base_capacity"]),
            source_system="minco_reference_case",
            source_mode=SourceMode.SYNTHETIC,
            metadata={"operation": "set", "evidence": "bundled_reference_capacity"},
        ))

    # Convert daily cohort arrivals into deterministic timestamps spread over the day.
    arrivals = instance.arrivals.df.sort_values(["day", "hospital_id", "cohort"]).reset_index(drop=True)
    for idx, row in arrivals.iterrows():
        day = int(row["day"])
        count = int(round(float(row["arrivals"])))
        if count <= 0:
            continue
        day_start = start + timedelta(days=day)
        for k in range(count):
            minute = int(((k + 0.5) / count) * 24 * 60)
            ts = day_start + timedelta(minutes=minute)
            events.append(HospitalEvent(
                event_timestamp=ts,
                hospital_id=str(row["hospital_id"]),
                event_type=EventType.ARRIVAL,
                patient_pathway_or_cohort=str(row["cohort"]),
                resource="ED",
                quantity=1.0,
                source_system="minco_reference_case",
                source_mode=SourceMode.SYNTHETIC,
                patient_id=f"SYN-{day:02d}-{row['hospital_id']}-{row['cohort']}-{k:04d}",
                metadata={"day": day, "synthetic": True},
            ))
    return sorted(events, key=lambda e: (e.event_timestamp, e.event_id))


def events_to_frame(events: list[HospitalEvent]) -> pd.DataFrame:
    rows = []
    for event in events:
        row = event.model_dump(mode="json")
        row["metadata"] = str(row["metadata"])
        rows.append(row)
    return pd.DataFrame(rows)
