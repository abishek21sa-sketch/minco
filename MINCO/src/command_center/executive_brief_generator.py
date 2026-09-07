from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from src.command_center.alert_engine import Alert, highest_alert_level


def generate_executive_brief(
    state: Dict[str, float],
    alerts: List[Alert],
    scenario_label: str = "current network state",
) -> str:
    """
    Produce a short Markdown executive brief summarizing current state,
    alerts, and recommended action -- written for a COO reading it during a
    morning operations huddle, not an engineer reading a log file.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    overall_level = highest_alert_level(alerts)

    level_language = {
        "RED": "**RED -- immediate attention required**",
        "YELLOW": "**YELLOW -- monitor closely**",
        "GREEN": "**GREEN -- stable**",
    }[overall_level]

    lines: List[str] = []
    lines.append("# Meridian Health Network -- Executive Brief")
    lines.append(f"_{timestamp}_")
    lines.append("")
    lines.append(f"## Overall Status: {level_language}")
    lines.append(f"Scenario: {scenario_label}")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append(f"- Unsafe excess: {state.get('total_unsafe_excess', 0.0):.2f}")
    lines.append(f"- Max utilization: {state.get('max_utilization_ratio', 0.0):.3f}")
    lines.append(f"- Blocked arrivals: {state.get('total_blocked_arrivals', 0.0):.2f}")
    lines.append(f"- Overflow excess: {state.get('total_overflow_excess', 0.0):.2f}")
    lines.append("")
    lines.append("## Alerts")
    for alert in alerts:
        lines.append(f"- **[{alert.level}]** ({alert.trigger}) {alert.message}")
    lines.append("")
    lines.append("## Recommended Action")
    if overall_level == "RED":
        lines.append(
            "Activate surge capacity and prioritize transfer-enabled coordination immediately. "
            "Escalate to operations leadership."
        )
    elif overall_level == "YELLOW":
        lines.append(
            "Maintain transfer-enabled coordination under the optimized_network policy; "
            "monitor ICU utilization and overflow closely over the next reporting cycle."
        )
    else:
        lines.append(
            "No intervention required. Continue standard coordinated operation under the "
            "optimized_network policy."
        )
    lines.append("")
    lines.append("---")
    lines.append(
        "_Generated live by the Meridian Command Center pipeline. Solved as a network "
        f"optimization model in {state.get('solve_runtime_seconds', 0.0):.2f}s "
        f"({int(state.get('n_replications', 0))} stochastic replications)._"
    )

    return "\n".join(lines)