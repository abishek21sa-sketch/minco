from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from src.hospital_events.models import HospitalEvent
from src.state_reconstruction.reconstructor import NetworkState, reconstruct_network_state


@dataclass(frozen=True)
class ReplayFrame:
    index: int
    as_of: datetime
    state: NetworkState


def build_replay_frames(
    events: Iterable[HospitalEvent],
    *,
    step: timedelta = timedelta(hours=6),
) -> list[ReplayFrame]:
    ordered = sorted(events, key=lambda e: (e.event_timestamp, e.event_id))
    if not ordered:
        return []
    if step.total_seconds() <= 0:
        raise ValueError("replay step must be positive")
    start = ordered[0].event_timestamp
    end = ordered[-1].event_timestamp
    frames: list[ReplayFrame] = []
    as_of = start
    index = 0
    while as_of < end:
        frames.append(ReplayFrame(index=index, as_of=as_of, state=reconstruct_network_state(ordered, as_of=as_of)))
        index += 1
        as_of += step
    frames.append(ReplayFrame(index=index, as_of=end, state=reconstruct_network_state(ordered, as_of=end)))
    return frames
