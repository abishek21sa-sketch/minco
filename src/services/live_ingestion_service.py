"""Governed adapter boundary for external hospital event streams."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.contracts.common import ExecutionContext, new_correlation_id, utc_now
from src.contracts.event_ingestion import (
    EventIngestionRequest,
    EventIngestionResponse,
    EventIngestionStatusResponse,
)
from src.hospital_events.models import SourceMode
from src.services.operational_history_service import OperationalHistoryService


MAX_EVENT_AGE = timedelta(hours=24)
MAX_CLOCK_SKEW = timedelta(minutes=5)


class LiveEventIngestionService:
    """Validate, deduplicate, and append external events without executing actions."""

    service_name = "live_event_ingestion_service"

    def __init__(self, operational_history: OperationalHistoryService) -> None:
        self.operational_history = operational_history

    @staticmethod
    def _policy() -> dict[str, Any]:
        return {
            "max_batch_size": 500,
            "max_event_age_hours": 24,
            "max_future_clock_skew_minutes": 5,
            "deduplication_key": "event_id",
            "required_source_mode": "external_stream",
            "patient_identifiers_accepted": False,
        }

    @staticmethod
    def _freshness(latest_ingested: datetime | None, now: datetime) -> str:
        if latest_ingested is None:
            return "UNAVAILABLE"
        age = max(0.0, (now - latest_ingested).total_seconds())
        return "CURRENT" if age <= MAX_EVENT_AGE.total_seconds() else "STALE"

    def status(self, context: ExecutionContext | None = None) -> EventIngestionStatusResponse:
        context = context or ExecutionContext(
            correlation_id=new_correlation_id(), source="service"
        )
        generated_at = utc_now()
        try:
            events = self.operational_history.event_store().events_between()
            external = [
                event for event in events
                if str(getattr(event.source_mode, "value", event.source_mode)) == SourceMode.EXTERNAL_STREAM.value
            ]
            latest_event = max((event.event_timestamp for event in external), default=None)
            latest_ingested = max((event.ingested_at for event in external), default=None)
            return EventIngestionStatusResponse(
                generated_at=generated_at,
                correlation_id=context.correlation_id,
                gateway_available=True,
                live_feed_connected=bool(external),
                data_mode="EXTERNAL_STREAM" if external else "DISCONNECTED",
                source_systems=sorted({event.source_system for event in external}),
                accepted_event_count=len(external),
                latest_event_timestamp=latest_event,
                latest_ingested_at=latest_ingested,
                freshness_status=self._freshness(latest_ingested, generated_at),
                validation_policy=self._policy(),
                warnings=([] if external else ["No external ADT/EHR stream has been connected to this reference package."]),
            )
        except Exception as exc:
            return EventIngestionStatusResponse(
                generated_at=generated_at,
                correlation_id=context.correlation_id,
                gateway_available=False,
                live_feed_connected=False,
                data_mode="DISCONNECTED",
                source_systems=[],
                accepted_event_count=0,
                freshness_status="UNAVAILABLE",
                validation_policy=self._policy(),
                warnings=[f"Live event gateway unavailable: {type(exc).__name__}: {exc}"],
            )

    def ingest(
        self,
        request: EventIngestionRequest,
        context: ExecutionContext | None = None,
    ) -> EventIngestionResponse:
        context = context or ExecutionContext(
            correlation_id=new_correlation_id(), source="service"
        )
        generated_at = utc_now()
        rejected: list[dict[str, Any]] = []
        now = generated_at
        for index, event in enumerate(request.events):
            detail: str | None = None
            if event.source_mode != SourceMode.EXTERNAL_STREAM:
                detail = "event source_mode must be external_stream"
            elif event.source_system != request.source_system:
                detail = "event source_system must match the batch source_system"
            elif event.patient_id:
                detail = "patient_id is not accepted by this gateway; use an approved de-identified cohort boundary"
            elif event.event_timestamp > now + MAX_CLOCK_SKEW:
                detail = "event_timestamp is beyond the allowed future clock-skew window"
            elif now - event.event_timestamp > MAX_EVENT_AGE:
                detail = "event_timestamp is older than the 24-hour live-ingestion window"
            if detail:
                rejected.append({"index": index, "event_id": event.event_id, "code": "validation_error", "detail": detail})

        valid = [
            event for index, event in enumerate(request.events)
            if not any(item["index"] == index for item in rejected)
        ]
        try:
            store = self.operational_history.event_store()
            existing_ids = {event.event_id for event in store.events_between()}
            duplicates = [event for event in valid if event.event_id in existing_ids]
            new_events = [event for event in valid if event.event_id not in existing_ids]
            inserted = store.append(new_events)
            accepted_ids = [event.event_id for event in new_events[:inserted]]
            latest = max((event.event_timestamp for event in new_events), default=None)
            duplicate_count = len(duplicates)
            rejected_count = len(rejected)
            status = (
                "REJECTED" if not accepted_ids and not duplicates and rejected_count else
                "PARTIAL" if rejected_count else
                "ACCEPTED"
            )
            return EventIngestionResponse(
                generated_at=generated_at,
                correlation_id=context.correlation_id,
                source_system=request.source_system,
                status=status,
                accepted_count=len(accepted_ids),
                duplicate_count=duplicate_count,
                rejected_count=rejected_count,
                accepted_event_ids=accepted_ids,
                rejected_events=rejected,
                latest_event_timestamp=latest,
                freshness_status=self._freshness(max((event.ingested_at for event in new_events), default=None), generated_at),
                live_feed_connected=bool(accepted_ids or duplicates),
                validation_policy=self._policy(),
            )
        except Exception as exc:
            return EventIngestionResponse(
                generated_at=generated_at,
                correlation_id=context.correlation_id,
                source_system=request.source_system,
                status="UNAVAILABLE",
                accepted_count=0,
                duplicate_count=0,
                rejected_count=len(rejected),
                rejected_events=[*rejected, {"code": "gateway_unavailable", "detail": f"{type(exc).__name__}: {exc}"}],
                freshness_status="UNAVAILABLE",
                validation_policy=self._policy(),
            )
