from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config.paths import DATA_DIR, RESULTS_DIR
from src.data_runtime.memory_store import MemoryEventStore
from src.hospital_events.synthetic_replay import build_reference_event_replay
from src.scenarios.scaled_replay import build_scaled_event_replay


def make_event_store(
    *,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
) -> tuple[Any, str, list[str]]:
    """Return the best configured Phase-2 event store.

    ``MINCO_EVENT_STORE=memory`` forces the dependency-free adapter. Otherwise
    DuckDB is preferred and the synthetic replay is seeded only when the lake is
    empty. This keeps deterministic demo mode separate from real file ingestion.
    """
    requested = os.getenv("MINCO_EVENT_STORE", "duckdb").strip().lower()
    replay_profile = os.getenv("MINCO_REPLAY_PROFILE", "enterprise").strip().lower()
    min_events = int(os.getenv("MINCO_REPLAY_MIN_EVENTS", "100000"))
    operating_mode = os.getenv("MINCO_OPERATING_MODE", "normal").strip().lower()
    n_hospitals = int(os.getenv("MINCO_REPLAY_HOSPITALS", "120"))
    public_reference_csv = DATA_DIR / "public_reference" / "cms_hospital_general_information.csv"
    warnings: list[str] = []
    if requested != "memory":
        try:
            from src.data_runtime.duckdb_store import DuckDBEventStore
            store = DuckDBEventStore(results_dir / "operational_lake" / "minco_events.duckdb")
            if replay_profile in {"enterprise", "scaled", "research"}:
                if store.count < min_events:
                    store.append(build_scaled_event_replay(
                        n_hospitals=n_hospitals,
                        minimum_events=min_events,
                        operating_mode=operating_mode,
                        public_reference_csv=public_reference_csv,
                    ))
            elif store.count == 0:
                store.append(build_reference_event_replay(data_dir=data_dir))
            try:
                store.export_parquet(results_dir / "operational_lake" / "hospital_events.parquet")
            except Exception as exc:
                warnings.append(f"Parquet export unavailable: {exc}")
            return store, "DUCKDB_PARQUET", warnings
        except Exception as exc:
            warnings.append(f"DuckDB event lake unavailable; using in-memory replay adapter: {exc}")
    store = MemoryEventStore()
    # Keep the explicit memory adapter lightweight and backward-compatible for
    # unit tests. Enterprise replay uses DuckDB so a 100k+ event lake is not
    # copied into every in-process test or gRPC fixture.
    store.append(build_reference_event_replay(data_dir=data_dir))
    return store, "MEMORY_SYNTHETIC_REPLAY", warnings
