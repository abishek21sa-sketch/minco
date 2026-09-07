"""Public-data reference adapters used to calibrate synthetic facilities.

Only public aggregate/provider metadata is used here. CMS facility metadata is
not a substitute for ADT/EHR feeds and is never treated as observed capacity,
occupancy, clinical outcomes or patient-level data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CMS_HOSPITAL_GENERAL_INFORMATION_DATASET = "xubh-q36u"
CMS_HOSPITAL_GENERAL_INFORMATION_API = (
    "https://data.cms.gov/provider-data/api/1/datastore/query/"
    f"{CMS_HOSPITAL_GENERAL_INFORMATION_DATASET}/0?limit=10000"
)
CDC_RESP_NET_SOURCE = "https://www.cdc.gov/resp-net/dashboard/index.html"
CDC_FLUVIEW_SOURCE = "https://www.cdc.gov/fluview/overview/fluview-interactive.html"


@dataclass(frozen=True)
class FacilityReference:
    facilities: pd.DataFrame
    provenance: dict[str, Any]


def _tier_for_row(row_index: int, hospital_type: str) -> str:
    value = hospital_type.lower()
    if "critical access" in value or "children" in value:
        return "community"
    # Deterministic strata for the public directory: this is a synthetic size
    # assignment, not a claim about the real facility's service capability.
    bucket = row_index % 10
    if bucket in {0, 1}:
        return "tertiary"
    if bucket in {2, 3, 4, 5, 6}:
        return "regional"
    return "community"


def _synthetic_rows(n_hospitals: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(n_hospitals):
        hospital_id = f"SYN-H{index + 1:04d}"
        rows.append(
            {
                "hospital_id": hospital_id,
                "hospital_name": f"Meridian Research Facility {index + 1:04d}",
                "hospital_type": ("tertiary" if index % 10 < 2 else "regional" if index % 10 < 7 else "community"),
                "state": "SYN",
                "public_facility_id": None,
                "facility_source": "synthetic_facility_catalog",
                "capacity_source": "synthetic_calibration",
                "volume_factor": round(0.78 + ((index * 17) % 59) / 100, 3),
            }
        )
    return pd.DataFrame(rows)


def load_facility_reference(
    *,
    n_hospitals: int = 120,
    csv_path: str | Path | None = None,
    seed: int = 20260906,
) -> FacilityReference:
    """Create a deterministic facility catalog from CMS metadata when present.

    If the optional CSV snapshot has not been refreshed, a synthetic catalog is
    returned. The output schema is identical in both cases so offline tests and
    research runs do not depend on network access.
    """
    if n_hospitals < 1:
        raise ValueError("n_hospitals must be positive")
    path = Path(csv_path) if csv_path else None
    if path and path.exists():
        source = pd.read_csv(path)
        required = {"facility_id", "facility_name", "hospital_type"}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"CMS facility snapshot missing columns: {missing}")
        source = source.dropna(subset=["facility_id", "facility_name", "hospital_type"]).copy()
        source["facility_id"] = source["facility_id"].astype(str).str.strip()
        source = source.sort_values("facility_id").drop_duplicates("facility_id").head(n_hospitals)
        if source.empty:
            raise ValueError("CMS facility snapshot contains no usable facilities")
        rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(seed)
        for index, row in enumerate(source.to_dict("records")):
            hospital_type = str(row["hospital_type"])
            rows.append(
                {
                    "hospital_id": f"CMS-{str(row['facility_id'])}",
                    "hospital_name": str(row["facility_name"]).strip(),
                    "hospital_type": _tier_for_row(index, hospital_type),
                    "state": str(row.get("state", "")),
                    "public_facility_id": str(row["facility_id"]),
                    "facility_source": f"CMS Hospital General Information {CMS_HOSPITAL_GENERAL_INFORMATION_DATASET}",
                    "capacity_source": "synthetic_calibration_from_public_facility_strata",
                    "volume_factor": round(float(rng.uniform(0.78, 1.42)), 3),
                }
            )
        facilities = pd.DataFrame(rows)
        provenance = {
            "mode": "public_aggregate_calibrated_synthetic",
            "source": CMS_HOSPITAL_GENERAL_INFORMATION_API,
            "dataset_id": CMS_HOSPITAL_GENERAL_INFORMATION_DATASET,
            "snapshot_path": str(path.resolve()),
            "rows_used": int(len(facilities)),
            "capacity_claim": "No observed bed/occupancy capacity is imported; capacities are synthetic.",
            "related_sources": [CDC_RESP_NET_SOURCE, CDC_FLUVIEW_SOURCE],
        }
        return FacilityReference(facilities=facilities, provenance=provenance)

    facilities = _synthetic_rows(n_hospitals)
    return FacilityReference(
        facilities=facilities,
        provenance={
            "mode": "synthetic_facility_catalog",
            "source": "No local public snapshot found; deterministic synthetic facility strata used.",
            "rows_used": int(len(facilities)),
            "capacity_claim": "Synthetic capacities only.",
            "related_sources": [CDC_RESP_NET_SOURCE, CDC_FLUVIEW_SOURCE],
        },
    )
