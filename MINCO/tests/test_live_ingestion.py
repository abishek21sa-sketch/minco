from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.contracts.common import ExecutionContext, utc_now
from src.contracts.event_ingestion import EventIngestionRequest
from src.data_runtime.memory_store import MemoryEventStore
from src.hospital_events.models import EventType, HospitalEvent, SourceMode
from src.services.container import build_service_container


def _event(*, event_id: str, timestamp_offset: timedelta = timedelta(0)) -> HospitalEvent:
    now = utc_now()
    return HospitalEvent(
        event_id=event_id,
        event_timestamp=now + timestamp_offset,
        ingested_at=now,
        hospital_id="H1",
        event_type=EventType.ARRIVAL,
        patient_pathway_or_cohort="adult_medical",
        resource="ED",
        quantity=1.0,
        source_system="ADT_TEST",
        source_mode=SourceMode.EXTERNAL_STREAM,
    )


def _reference_store() -> MemoryEventStore:
    store = MemoryEventStore()
    store.append(
        [
            _event(event_id="reference-baseline").model_copy(
                update={"source_mode": SourceMode.SYNTHETIC}
            )
        ]
    )
    return store


def test_live_ingestion_validates_deduplicates_and_updates_shared_history(tmp_path) -> None:
    services = build_service_container(
        data_dir=tmp_path,
        results_dir=tmp_path,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
        event_store=_reference_store(),
    )
    request = EventIngestionRequest(
        source_system="ADT_TEST",
        events=[_event(event_id="external-001")],
    )

    first = services.live_ingestion.ingest(
        request,
        ExecutionContext(correlation_id="ingestion-test-01", source="test"),
    )
    duplicate = services.live_ingestion.ingest(request)
    stale = services.live_ingestion.ingest(
        EventIngestionRequest(
            source_system="ADT_TEST",
            events=[_event(event_id="external-stale", timestamp_offset=-timedelta(hours=25))],
        )
    )

    assert first.status == "ACCEPTED"
    assert first.accepted_count == 1
    assert first.live_feed_connected is True
    assert duplicate.duplicate_count == 1
    assert duplicate.accepted_count == 0
    assert stale.status == "REJECTED"
    assert stale.rejected_count == 1
    history = services.operational_history.snapshot()
    assert history.live_feed_connected is True
    assert "external_stream" in history.source_modes


def test_ingestion_api_exposes_disconnected_status_and_rejects_untrusted_provenance(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("MINCO_API_KEY", raising=False)
    services = build_service_container(
        data_dir=tmp_path,
        results_dir=tmp_path,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
        event_store=_reference_store(),
    )
    client = TestClient(create_app(services))

    status = client.get(
        "/v1/operational-events/status",
        headers={"X-Correlation-ID": "ingestion-api-01"},
    )
    payload = _event(event_id="file-batch-001").model_dump(mode="json")
    payload["source_mode"] = "file_batch"
    response = client.post(
        "/v1/operational-events/ingest",
        json={"source_system": "ADT_TEST", "events": [payload]},
        headers={"X-Correlation-ID": "ingestion-api-02"},
    )

    assert status.status_code == 200
    assert status.json()["data_mode"] == "DISCONNECTED"
    assert status.json()["live_feed_connected"] is False
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "ingestion-api-02"
    assert response.json()["status"] == "REJECTED"
    assert response.json()["rejected_events"][0]["code"] == "validation_error"


def test_live_ingestion_rejects_raw_patient_identifiers(tmp_path) -> None:
    services = build_service_container(
        data_dir=tmp_path,
        results_dir=tmp_path,
        db_path=tmp_path / "audit.db",
        manifest_dir=tmp_path / "manifests",
        event_store=_reference_store(),
    )
    event = _event(event_id="phi-001").model_copy(update={"patient_id": "patient-raw-001"})

    result = services.live_ingestion.ingest(
        EventIngestionRequest(source_system="ADT_TEST", events=[event])
    )

    assert result.status == "REJECTED"
    assert result.rejected_events[0]["detail"].startswith("patient_id is not accepted")
