"""DuckDB + Parquet operational event lake.

DuckDB is a Phase-2 optional dependency because the exact clean-extraction
builder may not have network access. On the Windows acceptance machine install
MINCO with the ``phase2`` extra and this becomes the primary historical/replay
store.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.hospital_events.models import HospitalEvent


DDL = """
CREATE TABLE IF NOT EXISTS hospital_events (
    event_id VARCHAR PRIMARY KEY,
    event_timestamp TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL,
    hospital_id VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    patient_pathway_or_cohort VARCHAR NOT NULL,
    resource VARCHAR NOT NULL,
    quantity DOUBLE NOT NULL,
    source_system VARCHAR NOT NULL,
    source_mode VARCHAR NOT NULL,
    unit_id VARCHAR,
    patient_id VARCHAR,
    from_hospital_id VARCHAR,
    to_hospital_id VARCHAR,
    metadata_json VARCHAR NOT NULL
)
"""


class DuckDBEventStore:
    def __init__(self, db_path: str | Path) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is unavailable. Install MINCO with `.[phase2]`.") from exc
        self._duckdb = duckdb
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(DDL)

    def _connect(self):
        return self._duckdb.connect(str(self.db_path))

    def append(self, events: Iterable[HospitalEvent]) -> int:
        rows = []
        for e in events:
            rows.append((
                e.event_id, e.event_timestamp, e.ingested_at, e.hospital_id, str(e.event_type),
                e.patient_pathway_or_cohort, e.resource, float(e.quantity), e.source_system,
                str(e.source_mode), e.unit_id, e.patient_id, e.from_hospital_id, e.to_hospital_id,
                json.dumps(e.metadata, sort_keys=True),
            ))
        if not rows:
            return 0
        columns = [
            "event_id", "event_timestamp", "ingested_at", "hospital_id", "event_type",
            "patient_pathway_or_cohort", "resource", "quantity", "source_system", "source_mode",
            "unit_id", "patient_id", "from_hospital_id", "to_hospital_id", "metadata_json",
        ]
        with self._connect() as con:
            before = int(con.execute("SELECT COUNT(*) FROM hospital_events").fetchone()[0])
            # Registering one DataFrame lets DuckDB vectorize the batch insert;
            # row-by-row executemany becomes unacceptably slow at 100k+ events.
            frame = pd.DataFrame(rows, columns=columns)
            con.register("minco_event_batch", frame)
            con.execute("INSERT OR IGNORE INTO hospital_events SELECT * FROM minco_event_batch")
            con.unregister("minco_event_batch")
            after = int(con.execute("SELECT COUNT(*) FROM hospital_events").fetchone()[0])
        return after - before

    def events_between(self, start: datetime | None = None, end: datetime | None = None) -> list[HospitalEvent]:
        clauses: list[str] = []
        params: list[object] = []
        if start is not None:
            clauses.append("event_timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("event_timestamp <= ?")
            params.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM hospital_events {where} ORDER BY event_timestamp, event_id"
        with self._connect() as con:
            columns = [d[0] for d in con.execute(query, params).description]
            rows = con.execute(query, params).fetchall()
        result: list[HospitalEvent] = []
        for values in rows:
            row = dict(zip(columns, values, strict=True))
            row["metadata"] = json.loads(row.pop("metadata_json"))
            result.append(HospitalEvent.model_validate(row))
        return result

    def latest_timestamp(self) -> datetime | None:
        with self._connect() as con:
            return con.execute("SELECT MAX(event_timestamp) FROM hospital_events").fetchone()[0]

    @property
    def count(self) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM hospital_events").fetchone()[0])

    def export_parquet(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        escaped = str(target.resolve()).replace("'", "''")
        with self._connect() as con:
            con.execute(f"COPY (SELECT * FROM hospital_events ORDER BY event_timestamp, event_id) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        return target
