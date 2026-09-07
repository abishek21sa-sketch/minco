"""Canonical operational KPI contract."""
from __future__ import annotations

from typing import Any, Mapping

from pydantic import Field

from src.contracts.common import StrictContract


class OperationalMetrics(StrictContract):
    total_unsafe_excess: float = 0.0
    total_overflow_excess: float = 0.0
    max_utilization_ratio: float = Field(default=0.0, ge=0.0)
    num_unsafe_rows: float = Field(default=0.0, ge=0.0)
    total_blocked_arrivals: float = Field(default=0.0, ge=0.0)
    n_replications: int = Field(default=1, ge=1)
    model_status: int | None = None
    solver_status_label: str | None = None
    objective_value: float | None = None
    mip_gap: float | None = Field(default=None, ge=0.0)
    best_bound: float | None = None
    is_mip: bool = False
    total_surge_activated: float = Field(default=0.0, ge=0.0)
    total_elective_rejected: float = Field(default=0.0, ge=0.0)
    total_icu_transfer_load: float = Field(default=0.0, ge=0.0)
    solve_runtime_seconds: float = Field(default=0.0, ge=0.0)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OperationalMetrics":
        return cls(
            total_unsafe_excess=float(values.get("total_unsafe_excess", 0.0)),
            total_overflow_excess=float(values.get("total_overflow_excess", 0.0)),
            max_utilization_ratio=float(values.get("max_utilization_ratio", 0.0)),
            num_unsafe_rows=float(values.get("num_unsafe_rows", 0.0)),
            total_blocked_arrivals=float(values.get("total_blocked_arrivals", 0.0)),
            n_replications=max(1, int(float(values.get("n_replications", 1)))),
            model_status=(
                None
                if values.get("model_status") is None
                else int(float(values["model_status"]))
            ),
            solver_status_label=(
                None if values.get("solver_status_label") is None else str(values["solver_status_label"])
            ),
            objective_value=(
                None if values.get("objective_value") is None else float(values["objective_value"])
            ),
            mip_gap=None if values.get("mip_gap") is None else float(values["mip_gap"]),
            best_bound=None if values.get("best_bound") is None else float(values["best_bound"]),
            is_mip=bool(values.get("is_mip", False)),
            total_surge_activated=float(values.get("total_surge_activated", 0.0)),
            total_elective_rejected=float(values.get("total_elective_rejected", 0.0)),
            total_icu_transfer_load=float(values.get("total_icu_transfer_load", 0.0)),
            solve_runtime_seconds=float(values.get("solve_runtime_seconds", 0.0)),
        )
