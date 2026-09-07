"""Optional Anthropic client adapter with family-level model resolution.

The API key is read only from ANTHROPIC_API_KEY. Model IDs may be pinned using
MINCO_CLAUDE_HAIKU_MODEL / MINCO_CLAUDE_SONNET_MODEL. If not pinned, the
adapter queries Anthropic's Models API and chooses the newest available model
whose display name/id matches the requested family.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any

from .router import RoutingDecision, route_claude_query


@dataclass(frozen=True)
class ClaudeResponse:
    text: str
    model: str
    route: RoutingDecision
    input_tokens: int | None
    output_tokens: int | None


def _client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Install the optional 'llm' dependency to use Claude") from exc
    return anthropic.Anthropic(api_key=api_key)


def _created_value(model: Any) -> str:
    value = getattr(model, "created_at", None)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def resolve_model_id(client: Any, family: str) -> str:
    family = family.lower()
    env_name = "MINCO_CLAUDE_SONNET_MODEL" if family == "sonnet" else "MINCO_CLAUDE_HAIKU_MODEL"
    pinned = os.getenv(env_name, "").strip()
    if pinned:
        return pinned
    response = client.models.list(limit=100)
    models = list(getattr(response, "data", response))
    candidates = []
    for model in models:
        model_id = str(getattr(model, "id", ""))
        display = str(getattr(model, "display_name", ""))
        if family in f"{model_id} {display}".lower():
            candidates.append(model)
    if not candidates:
        raise RuntimeError(f"No Anthropic {family!r} family model is available to this API key")
    chosen = max(candidates, key=_created_value)
    return str(getattr(chosen, "id"))


def ask_claude(
    query: str,
    *,
    system_prompt: str,
    evidence: str,
    tool_count: int = 0,
    requires_cross_module_reasoning: bool = False,
    sonnet_tokens_used_today: int = 0,
) -> ClaudeResponse:
    route = route_claude_query(
        query,
        tool_count=tool_count,
        requires_cross_module_reasoning=requires_cross_module_reasoning,
        sonnet_tokens_used_today=sonnet_tokens_used_today,
    )
    client = _client()
    model = resolve_model_id(client, route.model_family)
    response = client.messages.create(
        model=model,
        max_tokens=route.max_output_tokens,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": (
                "Use only the deterministic MINCO evidence below for engineering claims. "
                "Distinguish observed, predicted, simulated, and optimized quantities.\n\n"
                f"EVIDENCE:\n{evidence}\n\nQUERY:\n{query}"
            ),
        }],
    )
    text = "".join(getattr(block, "text", "") for block in getattr(response, "content", []))
    usage = getattr(response, "usage", None)
    return ClaudeResponse(
        text=text,
        model=model,
        route=route,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )
