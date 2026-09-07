# Canonical Hospital Event Contract

Every operational event requires:

- timezone-aware `event_timestamp`
- `hospital_id`
- `event_type`
- `patient_pathway_or_cohort`
- `resource`
- finite `quantity`
- `source_system`
- explicit `source_mode`

Supported event types include arrival, admission, bed request/assignment, transfer request/completion, discharge readiness/discharge, capacity changes, staffing changes and diversion.

`transfer_complete` requires distinct origin and destination hospitals. Capacity/staffing changes cannot have a zero quantity. The implementation is `src/hospital_events/models.py`.

The Phase-2 local event lake is implemented in `src/data_runtime/duckdb_store.py`. It stores canonical events transactionally in DuckDB and exports an ordered ZSTD-compressed Parquet replay asset. `src/data_runtime/polars_analytics.py` performs lazy Parquet analytics.

The bundled `data/sample_external/canonical_events.csv` demonstrates the external file boundary. It is demonstration data, not hospital data.
