from __future__ import annotations

import json
from typing import Any


def compact_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def build_review_evidence(*, state: Any, plan: dict[str, Any] | None, simulation: dict[str, Any] | None) -> str:
    sections = ["STATE=" + compact_json(state)]
    if plan is not None:
        sections.append("PLAN=" + compact_json(plan))
    if simulation is not None:
        sections.append("SIMULATION=" + compact_json(simulation))
    return "\n".join(sections)
