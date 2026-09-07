from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.config.dataclasses import HealthcareInstance


Arc = Tuple[str, str]


def build_allowed_transfer_lanes(
    instance: HealthcareInstance,
) -> pd.DataFrame:
    """
    Return the allowed transfer lanes from the instance.

    Keeps only rows with allowed == 1 and removes self-loops if present.
    """
    lanes = instance.transfer_lanes.df.copy()

    lanes = lanes[lanes["allowed"] == 1].copy()
    lanes = lanes[lanes["from_hospital"] != lanes["to_hospital"]].copy()

    lanes = lanes[
        ["from_hospital", "to_hospital", "transfer_capacity", "transfer_cost", "transfer_time"]
    ].drop_duplicates()

    lanes = lanes.sort_values(
        ["from_hospital", "to_hospital"]
    ).reset_index(drop=True)

    return lanes


def build_allowed_transfer_arcs(
    instance: HealthcareInstance,
) -> List[Arc]:
    """
    Return a sorted list of allowed directed transfer arcs.
    """
    lanes = build_allowed_transfer_lanes(instance)

    arcs: List[Arc] = [
        (str(row.from_hospital), str(row.to_hospital))
        for row in lanes.itertuples(index=False)
    ]

    return arcs


def build_transfer_capacity_lookup(
    instance: HealthcareInstance,
) -> Dict[Arc, float]:
    """Build the maximum modeled ICU transfer load per directed lane and day.

    ``allowed`` remains a binary topology switch. ``transfer_capacity`` is the
    continuous operational lane limit and may be scaled by scenario logic
    without changing whether the lane exists.
    """
    lanes = build_allowed_transfer_lanes(instance)
    return {
        (str(row.from_hospital), str(row.to_hospital)): float(row.transfer_capacity)
        for row in lanes.itertuples(index=False)
    }


def build_transfer_cost_lookup(
    instance: HealthcareInstance,
    include_generic_cost: bool = True,
) -> Dict[Arc, float]:
    """
    Build arc-specific transfer cost lookup.

    If include_generic_cost is True, adds the generic transfer penalty
    from costs.csv to each arc-specific transfer cost.
    """
    lanes = build_allowed_transfer_lanes(instance)

    generic_transfer_cost = 0.0
    if include_generic_cost:
        costs_df = instance.costs.df.copy()
        cost_lookup = dict(zip(costs_df["cost_name"], costs_df["value"]))
        generic_transfer_cost = float(cost_lookup.get("transfer", 0.0))

    lookup: Dict[Arc, float] = {}

    for row in lanes.itertuples(index=False):
        arc = (str(row.from_hospital), str(row.to_hospital))
        lookup[arc] = float(row.transfer_cost) + generic_transfer_cost

    return lookup


def build_outbound_arc_map(
    arcs: List[Arc],
) -> Dict[str, List[Arc]]:
    """
    Build hospital -> list of outbound arcs.
    """
    outbound: Dict[str, List[Arc]] = {}

    for i, j in arcs:
        outbound.setdefault(i, []).append((i, j))

    for hospital in outbound:
        outbound[hospital] = sorted(outbound[hospital])

    return outbound


def build_inbound_arc_map(
    arcs: List[Arc],
) -> Dict[str, List[Arc]]:
    """
    Build hospital -> list of inbound arcs.
    """
    inbound: Dict[str, List[Arc]] = {}

    for i, j in arcs:
        inbound.setdefault(j, []).append((i, j))

    for hospital in inbound:
        inbound[hospital] = sorted(inbound[hospital])

    return inbound