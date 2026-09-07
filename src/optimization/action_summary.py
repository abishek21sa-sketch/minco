"""Solver-output summaries exposed to decision services without importing Gurobi."""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

GUROBI_STATUS_LABELS = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    6: "CUTOFF",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
    14: "INPROGRESS",
    15: "USER_OBJ_LIMIT",
    16: "WORK_LIMIT",
    17: "MEM_LIMIT",
}


def _sum_column(table: Any, column: str) -> float:
    if not isinstance(table, pd.DataFrame) or table.empty or column not in table.columns:
        return 0.0
    return float(pd.to_numeric(table[column], errors="coerce").fillna(0.0).sum())


def summarize_policy_actions(policy_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate the solved operational decisions in a policy snapshot."""
    metrics = policy_snapshot.get("model_metrics")
    row: dict[str, Any] = {}
    if isinstance(metrics, pd.DataFrame) and not metrics.empty:
        row = metrics.iloc[0].to_dict()

    status = None if row.get("model_status") is None else int(row["model_status"])
    return {
        "total_surge_activated": _sum_column(policy_snapshot.get("surge"), "surge_activated"),
        "total_elective_rejected": _sum_column(
            policy_snapshot.get("elective_rejected"), "elective_rejected"
        ),
        "total_icu_transfer_load": _sum_column(
            policy_snapshot.get("icu_transfers"), "icu_transfer_load"
        ),
        "objective_value": (
            None if row.get("objective_value") is None else float(row["objective_value"])
        ),
        "model_status": status,
        "solver_status_label": None if status is None else GUROBI_STATUS_LABELS.get(status, "UNKNOWN"),
        "mip_gap": None if row.get("mip_gap") is None else float(row["mip_gap"]),
        "best_bound": None if row.get("best_bound") is None else float(row["best_bound"]),
        "is_mip": bool(row.get("is_mip", False)),
    }
