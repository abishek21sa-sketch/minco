from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Tuple


@dataclass
class Alert:
    level: str  # "RED", "YELLOW", "GREEN"
    trigger: str
    message: str
    timestamp: str


BLOCKED_ARRIVALS_THRESHOLD = 10.0


def evaluate_alerts(
    state: Dict[str, float],
    classify_fn: Callable[[float, float, float], Tuple[str, str, str]],
) -> List[Alert]:
    """
    Evaluate the current state against the SAME stress thresholds used
    throughout the dashboard (classify_stress_label from
    src.analysis.stress_classification), so alert levels here always agree
    with what the System Status box, AI Copilot, and Risk Agent report --
    not a separate, drifting set of thresholds.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    max_util = float(state.get("max_utilization_ratio", 0.0))
    overflow = float(state.get("total_overflow_excess", 0.0))
    unsafe_rows = float(state.get("num_unsafe_rows", 0.0))
    blocked_arrivals = float(state.get("total_blocked_arrivals", 0.0))

    icu_status, overflow_status, bottleneck = classify_fn(max_util, overflow, unsafe_rows)

    alerts: List[Alert] = []

    if icu_status == "Critical":
        alerts.append(
            Alert(
                level="RED",
                trigger="icu_utilization",
                message=f"ICU utilization at {max_util:.2f}x capacity (critical threshold: 1.70x). {bottleneck}.",
                timestamp=timestamp,
            )
        )
    elif icu_status == "Stressed":
        alerts.append(
            Alert(
                level="YELLOW",
                trigger="icu_utilization",
                message=f"ICU utilization at {max_util:.2f}x capacity (stressed threshold: 1.20x).",
                timestamp=timestamp,
            )
        )

    if overflow_status == "High":
        alerts.append(
            Alert(
                level="RED",
                trigger="overflow",
                message=f"Network overflow excess at {overflow:.2f} (high threshold: 8.0).",
                timestamp=timestamp,
            )
        )
    elif overflow_status == "Moderate":
        alerts.append(
            Alert(
                level="YELLOW",
                trigger="overflow",
                message=f"Network overflow excess at {overflow:.2f} (above zero).",
                timestamp=timestamp,
            )
        )

    if blocked_arrivals > BLOCKED_ARRIVALS_THRESHOLD:
        alerts.append(
            Alert(
                level="RED",
                trigger="blocked_arrivals",
                message=(
                    f"Forecast blocked arrivals at {blocked_arrivals:.2f} "
                    f"(threshold: {BLOCKED_ARRIVALS_THRESHOLD:.1f})."
                ),
                timestamp=timestamp,
            )
        )

    if not alerts:
        alerts.append(
            Alert(
                level="GREEN",
                trigger="none",
                message="All monitored thresholds within normal range.",
                timestamp=timestamp,
            )
        )

    return alerts


def highest_alert_level(alerts: List[Alert]) -> str:
    levels = {a.level for a in alerts}
    if "RED" in levels:
        return "RED"
    if "YELLOW" in levels:
        return "YELLOW"
    return "GREEN"