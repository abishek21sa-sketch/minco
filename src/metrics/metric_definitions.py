from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CapacityMetrics:
    total_unsafe_excess: float
    total_overflow_excess: float
    total_surge_gap: float
    max_utilization_ratio: float
    num_unsafe_rows: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
