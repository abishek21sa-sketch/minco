from __future__ import annotations

import argparse
from pathlib import Path

from src.data_runtime.csv_ingestion import load_canonical_event_csv
from src.data_runtime.duckdb_store import DuckDBEventStore
from src.config.paths import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest canonical MINCO hospital events into the DuckDB event lake")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--db", type=Path, default=RESULTS_DIR / "operational_lake" / "minco_events.duckdb")
    parser.add_argument("--parquet", type=Path, default=RESULTS_DIR / "operational_lake" / "hospital_events.parquet")
    args = parser.parse_args()
    events = load_canonical_event_csv(args.csv)
    store = DuckDBEventStore(args.db)
    inserted = store.append(events)
    parquet = store.export_parquet(args.parquet)
    print({"status":"passed","input_events":len(events),"inserted":inserted,"event_count":store.count,"parquet":str(parquet)})


if __name__ == "__main__":
    main()
