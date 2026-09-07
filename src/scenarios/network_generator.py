from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


TIER_PROFILES = {
    "tertiary": {
        "hospital_type": "tertiary",
        "icu_capacity": 40,
        "ward_capacity": 140,
        "icu_surge": 8,
        "ward_surge": 20,
        "icu_safe": 0.85,
        "ward_safe": 0.90,
        "arrivals": {
            "c1": [12, 13, 12, 14, 13, 12, 15],
            "c2": [8, 8, 9, 9, 8, 8, 10],
            "c3": [3, 3, 3, 4, 3, 3, 4],
            "c4": [5, 5, 6, 6, 5, 5, 6],
        },
        "elective_min": 2,
        "elective_max": 8,
    },
    "regional": {
        "hospital_type": "regional",
        "icu_capacity": 20,
        "ward_capacity": 90,
        "icu_surge": 5,
        "ward_surge": 12,
        "icu_safe": 0.85,
        "ward_safe": 0.90,
        "arrivals": {
            "c1": [9, 9, 10, 10, 9, 9, 11],
            "c2": [6, 6, 6, 7, 6, 6, 7],
            "c3": [2, 2, 2, 2, 2, 2, 2],
            "c4": [3, 3, 3, 3, 3, 3, 4],
        },
        "elective_min": 1,
        "elective_max": 5,
    },
    "community": {
        "hospital_type": "community",
        "icu_capacity": 10,
        "ward_capacity": 60,
        "icu_surge": 2,
        "ward_surge": 8,
        "icu_safe": 0.85,
        "ward_safe": 0.90,
        "arrivals": {
            "c1": [6, 6, 6, 7, 6, 6, 7],
            "c2": [4, 4, 4, 4, 4, 4, 5],
            "c3": [1, 1, 1, 1, 1, 1, 1],
            "c4": [2, 2, 2, 2, 2, 2, 2],
        },
        "elective_min": 0,
        "elective_max": 3,
    },
}

DAYS = list(range(1, 8))
COHORTS = ["c1", "c2", "c3", "c4"]


def _assign_tiers(n_hospitals: int, cluster_size: int) -> List[Tuple[str, str]]:
    """
    Return [(hospital_id, tier), ...] for n_hospitals hospitals grouped into
    clusters of cluster_size. Each cluster gets exactly one tertiary hub; the
    rest of the cluster splits into a couple of regional hospitals with the
    remainder as community hospitals -- mirroring the original 1 tertiary /
    1 regional / 1 community composition, just repeated at scale.
    """
    assignments: List[Tuple[str, str]] = []
    h_index = 1
    n_clusters = max(1, -(-n_hospitals // cluster_size))  # ceil division

    remaining = n_hospitals
    for _ in range(n_clusters):
        this_cluster_size = min(cluster_size, remaining)
        if this_cluster_size <= 0:
            break
        for slot in range(this_cluster_size):
            hospital_id = f"H{h_index}"
            if slot == 0:
                tier = "tertiary"
            elif slot <= 2:
                tier = "regional"
            else:
                tier = "community"
            assignments.append((hospital_id, tier))
            h_index += 1
        remaining -= this_cluster_size

    return assignments


def generate_synthetic_network(
    n_hospitals: int,
    cluster_size: int = 7,
) -> Dict[str, pd.DataFrame]:
    """
    Generate a full synthetic Meridian-style network instance (as a dict of
    DataFrames matching the base_instance schema) for n_hospitals hospitals.

    Hospitals are grouped into hub-and-spoke clusters of cluster_size: each
    cluster has one tertiary hub that spokes transfer into, and cluster hubs
    are connected to each other in a ring backbone. This keeps the number of
    transfer lanes roughly linear in n_hospitals instead of the N^2 blowup a
    fully-connected network would produce, which is both more realistic and
    keeps the MILP solvable at scale.
    """
    assignments = _assign_tiers(n_hospitals, cluster_size)

    hospitals_rows = []
    capacities_rows = []
    surge_rows = []
    safe_rows = []
    arrivals_rows = []
    elective_rows = []

    clusters: Dict[int, List[str]] = {}
    tier_by_id: Dict[str, str] = {}

    for idx, (hospital_id, tier) in enumerate(assignments):
        cluster_idx = idx // cluster_size
        clusters.setdefault(cluster_idx, []).append(hospital_id)
        tier_by_id[hospital_id] = tier

        profile = TIER_PROFILES[tier]

        hospitals_rows.append(
            {
                "hospital_id": hospital_id,
                "hospital_name": f"Meridian {tier.capitalize()} {hospital_id}",
                "hospital_type": profile["hospital_type"],
            }
        )

        capacities_rows.append({"hospital_id": hospital_id, "resource": "ICU", "base_capacity": profile["icu_capacity"]})
        capacities_rows.append({"hospital_id": hospital_id, "resource": "Ward", "base_capacity": profile["ward_capacity"]})

        surge_rows.append({"hospital_id": hospital_id, "resource": "ICU", "max_surge": profile["icu_surge"]})
        surge_rows.append({"hospital_id": hospital_id, "resource": "Ward", "max_surge": profile["ward_surge"]})

        safe_rows.append({"hospital_id": hospital_id, "resource": "ICU", "safe_utilization": profile["icu_safe"]})
        safe_rows.append({"hospital_id": hospital_id, "resource": "Ward", "safe_utilization": profile["ward_safe"]})

        for cohort in COHORTS:
            day_values = profile["arrivals"][cohort]
            for day, value in zip(DAYS, day_values):
                arrivals_rows.append({"day": day, "hospital_id": hospital_id, "cohort": cohort, "arrivals": value})

        for day in DAYS:
            elective_rows.append(
                {
                    "day": day,
                    "hospital_id": hospital_id,
                    "cohort": "c3",
                    "min_elective": profile["elective_min"],
                    "max_elective": profile["elective_max"],
                }
            )

    transfer_rows = []

    def _add_lane(frm: str, to: str, cost: float, time: int) -> None:
        source_profile = TIER_PROFILES[tier_by_id[frm]]
        # Synthetic planning assumption: each directed lane can carry up to
        # 20% of the source hospital's nominal ICU beds per day.
        transfer_capacity = max(1.0, round(0.20 * float(source_profile["icu_capacity"]), 3))
        transfer_rows.append(
            {
                "from_hospital": frm,
                "to_hospital": to,
                "allowed": 1,
                "transfer_capacity": transfer_capacity,
                "transfer_cost": cost,
                "transfer_time": time,
            }
        )

    cluster_hubs: List[str] = []
    for members in clusters.values():
        hub = next((h for h in members if tier_by_id[h] == "tertiary"), members[0])
        cluster_hubs.append(hub)
        for member in members:
            if member == hub:
                continue
            _add_lane(member, hub, cost=15.0, time=1)
            _add_lane(hub, member, cost=12.0, time=1)

    if len(cluster_hubs) > 1:
        for i, hub in enumerate(cluster_hubs):
            next_hub = cluster_hubs[(i + 1) % len(cluster_hubs)]
            _add_lane(hub, next_hub, cost=22.0, time=2)
            _add_lane(next_hub, hub, cost=22.0, time=2)

    return {
        "hospitals": pd.DataFrame(hospitals_rows),
        "capacities": pd.DataFrame(capacities_rows),
        "surge_caps": pd.DataFrame(surge_rows),
        "safe_thresholds": pd.DataFrame(safe_rows),
        "transfer_lanes": pd.DataFrame(transfer_rows),
        "arrivals": pd.DataFrame(arrivals_rows),
        "elective_bounds": pd.DataFrame(elective_rows),
    }


def write_synthetic_instance(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    costs_source: Path,
    transitions_source_dir: Path,
) -> None:
    """
    Write generated tables to output_dir/base_instance/*.csv, and copy the
    (network-size-independent) costs.csv and cohort transition matrices
    unchanged from the original base instance.
    """
    base_dir = output_dir / "base_instance"
    base_dir.mkdir(parents=True, exist_ok=True)
    transitions_dir = output_dir / "transitions"
    transitions_dir.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        df.to_csv(base_dir / f"{name}.csv", index=False)

    shutil.copy(costs_source, base_dir / "costs.csv")

    for src_file in transitions_source_dir.glob("*.csv"):
        shutil.copy(src_file, transitions_dir / src_file.name)