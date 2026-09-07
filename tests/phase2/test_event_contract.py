from datetime import UTC, datetime

import pytest

from src.hospital_events.models import EventType, HospitalEvent


def test_canonical_event_requires_timezone_and_transfer_semantics():
    with pytest.raises(ValueError):
        HospitalEvent(
            event_timestamp=datetime(2026, 1, 1),
            hospital_id="H1",
            event_type="arrival",
            patient_pathway_or_cohort="c1",
            resource="ED",
            source_system="x",
        )
    with pytest.raises(ValueError):
        HospitalEvent(
            event_timestamp=datetime.now(UTC),
            hospital_id="H1",
            event_type=EventType.TRANSFER_COMPLETE,
            patient_pathway_or_cohort="c1",
            resource="ICU",
            source_system="x",
        )
    event = HospitalEvent(
        event_timestamp=datetime.now(UTC),
        hospital_id="H1",
        event_type=EventType.ARRIVAL,
        patient_pathway_or_cohort="c1",
        resource="ED",
        source_system="x",
    )
    assert event.quantity == 1
