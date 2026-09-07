from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from pydantic import BaseModel, Field

from src.hospital_events.models import EventType, HospitalEvent


class ResourceState(BaseModel):
    hospital_id: str
    resource: str
    capacity: float = 0.0
    occupied: float = 0.0
    waiting: float = 0.0
    utilization: float | None = None


class HospitalState(BaseModel):
    hospital_id: str
    resources: list[ResourceState]
    arrivals_seen: float = 0.0
    discharges_seen: float = 0.0
    transfers_in: float = 0.0
    transfers_out: float = 0.0


class NetworkState(BaseModel):
    as_of: datetime
    latest_event_timestamp: datetime | None
    event_count: int
    freshness_seconds: float | None
    hospitals: list[HospitalState]
    evidence_mode: str = "OBSERVED_OR_REPLAYED_EVENTS"
    warnings: list[str] = Field(default_factory=list)


def reconstruct_network_state(events: Iterable[HospitalEvent], *, as_of: datetime | None = None) -> NetworkState:
    ordered = sorted(events, key=lambda e: (e.event_timestamp, e.event_id))
    if as_of is None:
        as_of = ordered[-1].event_timestamp if ordered else datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    ordered = [e for e in ordered if e.event_timestamp <= as_of]

    capacities: dict[tuple[str, str], float] = {}
    occupied: dict[tuple[str, str], float] = {}
    waiting: dict[tuple[str, str], float] = {}
    summary: dict[str, dict[str, float]] = {}

    def s(h: str) -> dict[str, float]:
        return summary.setdefault(h, {"arrivals_seen": 0.0, "discharges_seen": 0.0, "transfers_in": 0.0, "transfers_out": 0.0})

    for e in ordered:
        key = (e.hospital_id, e.resource)
        if e.event_type == EventType.CAPACITY_CHANGE:
            operation = str(e.metadata.get("operation", "delta")).lower()
            capacities[key] = float(e.quantity) if operation == "set" else capacities.get(key, 0.0) + float(e.quantity)
        elif e.event_type == EventType.ARRIVAL:
            s(e.hospital_id)["arrivals_seen"] += e.quantity
            waiting[key] = waiting.get(key, 0.0) + e.quantity
        elif e.event_type == EventType.ADMISSION:
            occupied[key] = occupied.get(key, 0.0) + e.quantity
        elif e.event_type == EventType.BED_REQUEST:
            waiting[key] = waiting.get(key, 0.0) + e.quantity
        elif e.event_type == EventType.BED_ASSIGNED:
            waiting[key] = max(0.0, waiting.get(key, 0.0) - e.quantity)
            occupied[key] = occupied.get(key, 0.0) + e.quantity
        elif e.event_type == EventType.DISCHARGE:
            occupied[key] = max(0.0, occupied.get(key, 0.0) - e.quantity)
            s(e.hospital_id)["discharges_seen"] += e.quantity
        elif e.event_type == EventType.TRANSFER_COMPLETE:
            origin = e.from_hospital_id or e.hospital_id
            destination = e.to_hospital_id or e.hospital_id
            s(origin)["transfers_out"] += e.quantity
            s(destination)["transfers_in"] += e.quantity
            occupied[(origin, e.resource)] = max(0.0, occupied.get((origin, e.resource), 0.0) - e.quantity)
            occupied[(destination, e.resource)] = occupied.get((destination, e.resource), 0.0) + e.quantity

    resource_keys = set(capacities) | set(occupied) | set(waiting)
    hospitals = sorted({h for h, _ in resource_keys} | set(summary))
    hospital_states: list[HospitalState] = []
    for hospital in hospitals:
        resources = sorted({r for h, r in set(capacities) | set(occupied) | set(waiting) if h == hospital})
        rs: list[ResourceState] = []
        for resource in resources:
            cap = max(0.0, capacities.get((hospital, resource), 0.0))
            occ = max(0.0, occupied.get((hospital, resource), 0.0))
            q = max(0.0, waiting.get((hospital, resource), 0.0))
            rs.append(ResourceState(
                hospital_id=hospital,
                resource=resource,
                capacity=cap,
                occupied=occ,
                waiting=q,
                utilization=(occ / cap if cap > 0 else None),
            ))
        vals = s(hospital)
        hospital_states.append(HospitalState(hospital_id=hospital, resources=rs, **vals))

    latest = max((e.event_timestamp for e in ordered), default=None)
    freshness = max(0.0, (as_of - latest).total_seconds()) if latest else None
    warnings: list[str] = []
    if latest is None:
        warnings.append("No operational events are available.")
    elif freshness is not None and freshness > 3600:
        warnings.append("Operational state is more than one hour behind the selected as-of time.")
    return NetworkState(
        as_of=as_of,
        latest_event_timestamp=latest,
        event_count=len(ordered),
        freshness_seconds=freshness,
        hospitals=hospital_states,
        warnings=warnings,
    )
