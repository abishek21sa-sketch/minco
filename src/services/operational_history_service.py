"""Operational event-lake and replay-state service for the enterprise API."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.paths import DATA_DIR, RESULTS_DIR
from src.contracts.common import ExecutionContext, REFERENCE_CASE_BOUNDARY, new_correlation_id, utc_now
from src.contracts.operational_history import OperationalHistoryResponse
from src.data_runtime.factory import make_event_store
from src.hospital_events.models import HospitalEvent
from src.hospital_events.synthetic_replay import build_reference_event_replay
from src.state_reconstruction.reconstructor import reconstruct_network_state


def _age_hours(now: datetime, observed: datetime | None) -> float | None:
    if observed is None:
        return None
    return round(max(0.0, (now - observed).total_seconds() / 3600.0), 2)


class OperationalHistoryService:
    """Expose canonical replay history with explicit freshness and provenance."""

    service_name = "operational_history_service"

    def __init__(
        self,
        *,
        data_dir: Path = DATA_DIR,
        results_dir: Path = RESULTS_DIR,
        store: Any | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self._store = store
        self._data_stack = store.__class__.__name__ if store is not None else None
        self._store_warnings: list[str] = []

    def _get_store(self) -> Any:
        if self._store is None:
            self._store, self._data_stack, self._store_warnings = make_event_store(
                data_dir=self.data_dir,
                results_dir=self.results_dir,
            )
        elif getattr(self._store, "count", 0) == 0:
            self._store.append(build_reference_event_replay(data_dir=self.data_dir))
        return self._store

    def event_store(self) -> Any:
        """Return the shared event store used by history and ingestion services."""
        return self._get_store()

    def snapshot(self, context: ExecutionContext | None = None) -> OperationalHistoryResponse:
        context = context or ExecutionContext(
            correlation_id=new_correlation_id(), source="service"
        )
        generated_at = utc_now()
        try:
            store = self._get_store()
            events: list[HospitalEvent] = store.events_between()
            latest_event = max((event.event_timestamp for event in events), default=None)
            latest_ingested = max((event.ingested_at for event in events), default=None)
            as_of = latest_event or generated_at
            reconstructed = reconstruct_network_state(events, as_of=as_of)
            source_modes = sorted(
                {str(getattr(event.source_mode, "value", event.source_mode)) for event in events}
            )
            live_feed_connected = "external_stream" in source_modes
            replay_only = bool(source_modes) and set(source_modes).issubset(
                {"synthetic", "historical_replay"}
            )
            age_hours = _age_hours(generated_at, latest_ingested or latest_event)
            freshness_status = (
                "UNAVAILABLE"
                if latest_event is None
                else "REPLAY_ONLY"
                if replay_only
                else "CURRENT"
                if age_hours is not None and age_hours <= 24
                else "STALE"
            )
            lake_dir = self.results_dir / "operational_lake"
            db_path = getattr(store, "db_path", None)
            parquet_path = lake_dir / "hospital_events.parquet"
            warnings = [*self._store_warnings, *reconstructed.warnings]
            if replay_only:
                warnings.append(
                    "Canonical events are synthetic historical replay data; they are not a live ADT/EHR feed."
                )
            return OperationalHistoryResponse(
                generated_at=generated_at,
                correlation_id=context.correlation_id,
                data_mode=(
                    "OPERATIONAL_EVENT_LAKE"
                    if self._data_stack == "DUCKDB_PARQUET"
                    else "SYNTHETIC_HISTORICAL_REPLAY"
                ),
                data_stack=str(self._data_stack or "UNAVAILABLE"),
                event_count=len(events),
                latest_event_timestamp=latest_event,
                latest_ingested_at=latest_ingested,
                as_of=as_of,
                freshness_status=freshness_status,
                freshness={
                    "latest_ingested_at": latest_ingested,
                    "age_hours": age_hours,
                    "target_hours": 24,
                    "note": (
                        "Replay history is available, but freshness is not equivalent to live operational data."
                        if replay_only
                        else "Event ingestion is within the 24-hour freshness target."
                    ),
                },
                event_type_counts=dict(
                    sorted(
                        Counter(
                            str(getattr(event.event_type, "value", event.event_type))
                            for event in events
                        ).items()
                    )
                ),
                source_modes=source_modes,
                hospital_ids=sorted({event.hospital_id for event in events}),
                live_feed_connected=live_feed_connected,
                reconstructed_state=reconstructed.model_dump(mode="json"),
                evidence={
                    "evidence_label": "REPLAYED",
                    "reference_case": "synthetic-meridian-network",
                    "source_modes": source_modes,
                    "duckdb_path": str(db_path) if db_path is not None else None,
                    "parquet_path": str(parquet_path) if parquet_path.exists() else None,
                    "claim_boundary": REFERENCE_CASE_BOUNDARY,
                },
                warnings=warnings,
            )
        except Exception as exc:
            return OperationalHistoryResponse(
                generated_at=generated_at,
                correlation_id=context.correlation_id,
                data_mode="UNAVAILABLE",
                data_stack="UNAVAILABLE",
                event_count=0,
                as_of=generated_at,
                freshness_status="UNAVAILABLE",
                freshness={
                    "latest_ingested_at": None,
                    "age_hours": None,
                    "target_hours": 24,
                    "note": "Operational event history could not be loaded.",
                },
                event_type_counts={},
                source_modes=[],
                hospital_ids=[],
                reconstructed_state={},
                evidence={
                    "evidence_label": "UNAVAILABLE",
                    "reference_case": "synthetic-meridian-network",
                },
                warnings=[f"Operational history unavailable: {type(exc).__name__}: {exc}"],
            )
