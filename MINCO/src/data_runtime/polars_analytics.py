from __future__ import annotations

from pathlib import Path
from typing import Any


def summarize_parquet_with_polars(path: str | Path) -> dict[str, Any]:
    """Lazy Polars summary over the operational Parquet lake."""
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("Polars is unavailable. Install MINCO with `.[phase2]`.") from exc
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    lf = pl.scan_parquet(path)
    by_type = (
        lf.group_by(["hospital_id", "event_type"])
        .agg(pl.len().alias("event_count"), pl.col("quantity").sum().alias("quantity_total"))
        .sort(["hospital_id", "event_type"])
        .collect()
    )
    return {
        "rows": int(lf.select(pl.len()).collect().item()),
        "hospital_event_summary": by_type.to_dicts(),
        "engine": "Polars lazy Parquet scan",
    }
