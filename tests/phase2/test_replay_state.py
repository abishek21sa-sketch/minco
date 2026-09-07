from datetime import timedelta
from itertools import pairwise

from src.data_runtime.memory_store import MemoryEventStore
from src.data_runtime.replay_engine import build_replay_frames
from src.hospital_events.synthetic_replay import build_reference_event_replay
from src.state_reconstruction.reconstructor import reconstruct_network_state


def test_reference_replay_reconstructs_causal_state_and_frames():
    events = build_reference_event_replay()
    assert len(events) > 450
    assert events == sorted(events, key=lambda e: (e.event_timestamp, e.event_id))
    store = MemoryEventStore()
    assert store.append(events) == len(events)
    assert store.append(events) == 0
    state = reconstruct_network_state(store.events_between())
    assert state.event_count == len(events)
    assert len(state.hospitals) == 3
    capacities = {
        (r.hospital_id, r.resource): r.capacity for h in state.hospitals for r in h.resources
    }
    assert capacities
    assert all(v >= 0 for v in capacities.values())
    frames = build_replay_frames(events, step=timedelta(hours=12))
    assert len(frames) > 10
    assert frames[0].state.event_count <= frames[-1].state.event_count
    assert all(a.as_of <= b.as_of for a, b in pairwise(frames))
