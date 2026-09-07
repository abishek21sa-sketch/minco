from __future__ import annotations

from datetime import datetime

from src.hospital_events.models import HospitalEvent


class MemoryEventStore:
    """Deterministic adapter used by mathematical tests and offline demo mode."""

    def __init__(self) -> None:
        self._events: dict[str, HospitalEvent] = {}

    def append(self, events: list[HospitalEvent]) -> int:
        before = len(self._events)
        for event in events:
            self._events.setdefault(event.event_id, event)
        return len(self._events) - before

    def events_between(self, start: datetime | None = None, end: datetime | None = None) -> list[HospitalEvent]:
        events = list(self._events.values())
        if start is not None:
            events = [e for e in events if e.event_timestamp >= start]
        if end is not None:
            events = [e for e in events if e.event_timestamp <= end]
        return sorted(events, key=lambda e: (e.event_timestamp, e.event_id))

    def latest_timestamp(self) -> datetime | None:
        return max((e.event_timestamp for e in self._events.values()), default=None)

    @property
    def count(self) -> int:
        return len(self._events)
