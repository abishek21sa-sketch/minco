from pathlib import Path

from src.data_runtime.csv_ingestion import load_canonical_event_csv


def test_sample_external_csv_maps_to_canonical_events():
    events = load_canonical_event_csv(Path("data/sample_external/canonical_events.csv"))
    assert len(events) == 9
    assert events[0].hospital_id == "H1"
    assert all(e.source_system == "external_demo" for e in events)


def test_duckdb_parquet_and_polars_contracts_are_explicit():
    duck = Path("src/data_runtime/duckdb_store.py").read_text(encoding="utf-8")
    polars = Path("src/data_runtime/polars_analytics.py").read_text(encoding="utf-8")
    assert "FORMAT PARQUET" in duck
    assert "COMPRESSION ZSTD" in duck
    assert "scan_parquet" in polars
    assert "group_by" in polars
