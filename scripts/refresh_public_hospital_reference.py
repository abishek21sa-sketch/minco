"""Refresh the optional CMS public hospital-directory snapshot.

This pulls provider metadata only. MINCO never treats this file as observed
capacity, occupancy, ADT, EHR or patient-level medical data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.scenarios.public_reference import CMS_HOSPITAL_GENERAL_INFORMATION_API


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/public_reference/cms_hospital_general_information.csv"),
    )
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    rows: list[dict] = []
    page_size = min(1000, max(1, args.limit))
    offset = 0
    while len(rows) < args.limit:
        page_url = (
            CMS_HOSPITAL_GENERAL_INFORMATION_API.split("?", 1)[0]
            + "?"
            + urlencode({"offset": offset, "limit": page_size})
        )
        request = Request(page_url, headers={"User-Agent": "MINCO-academic-research/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
        page_rows = payload.get("results", [])
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
        if len(page_rows) < page_size:
            break
    rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("CMS API returned no hospital rows")
    frame = pd.DataFrame(rows)
    required = {"facility_id", "facility_name", "hospital_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"CMS API response is missing required fields: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        {
            "status": "PASS",
            "rows": len(frame),
            "output": str(args.output),
            "source": CMS_HOSPITAL_GENERAL_INFORMATION_API,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
