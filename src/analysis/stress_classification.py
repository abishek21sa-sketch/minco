from __future__ import annotations

from typing import Tuple


def classify_stress_label(max_util: float, overflow: float, unsafe_rows: float) -> Tuple[str, str, str]:
    """
    Single source of truth for stress/risk classification thresholds, used by
    the dashboard's System Status box, the AI Copilot, the Risk Agent, and the
    Command Center's Alert Engine -- so "Critical" or "Stable" always means
    the same thing everywhere in the platform.
    """
    if max_util > 1.7:
        icu_status = "Critical"
    elif max_util > 1.2:
        icu_status = "Stressed"
    else:
        icu_status = "Stable"

    if overflow > 8:
        overflow_status = "High"
    elif overflow > 0:
        overflow_status = "Moderate"
    else:
        overflow_status = "Low"

    if unsafe_rows > 4:
        bottleneck = "Meridian Community ICU / Severe Network Stress"
    elif unsafe_rows > 2:
        bottleneck = "Local Capacity Stress"
    else:
        bottleneck = "No severe bottleneck detected"

    return icu_status, overflow_status, bottleneck


def classify_operating_regime(max_util: float) -> str:
    """
    Aligned with the same 1.2 / 1.5 thresholds classify_stress_label uses,
    so the regime label and the executive verdict never contradict each other.
    """
    if max_util < 1.2:
        return "stable regime"
    if max_util < 1.5:
        return "stressed regime"
    return "critical regime"