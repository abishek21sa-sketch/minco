from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.hospital_events.models import HospitalEvent

REQUIRED_COLUMNS = {
    "event_timestamp", "hospital_id", "event_type", "patient_pathway_or_cohort",
    "resource", "quantity", "source_system",
}


def load_canonical_event_csv(path: str | Path) -> list[HospitalEvent]:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical event CSV is missing required fields: {missing}")
    events: list[HospitalEvent] = []
    for row in frame.to_dict("records"):
        if "metadata" in row and isinstance(row["metadata"], str) and row["metadata"].strip():
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                row["metadata"] = {"raw_metadata": row["metadata"]}
        row["event_timestamp"] = pd.Timestamp(row["event_timestamp"]).to_pydatetime()
        if row["event_timestamp"].tzinfo is None:
            raise ValueError("event_timestamp values must include timezone offsets")
        row = {k: v for k, v in row.items() if not (isinstance(v, float) and pd.isna(v))}
        events.append(HospitalEvent.model_validate(row))
    return events
