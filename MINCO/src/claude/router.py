"""Cost-aware adaptive Claude routing for MINCO.

Routing is deterministic and auditable. It never asks Claude to decide which
Claude model should be used, avoiding an extra paid classification call.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re


@dataclass(frozen=True)
class RoutingDecision:
    tier: str  # "fast" (Haiku family) or "deep" (Sonnet family)
    complexity_score: int
    reason: tuple[str, ...]
    max_output_tokens: int
    model_family: str


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def route_claude_query(
    query: str,
    *,
    tool_count: int = 0,
    requires_cross_module_reasoning: bool = False,
    sonnet_tokens_used_today: int = 0,
) -> RoutingDecision:
    text = (query or "").strip().lower()
    if not text:
        raise ValueError("query must be nonempty")
    mode = os.getenv("MINCO_CLAUDE_ROUTING_MODE", "balanced").strip().lower()
    threshold = {"economy": 6, "balanced": 4, "quality": 3}.get(mode, 4)
    reasons: list[str] = []
    score = 0

    simple_markers = ["summarize", "status", "what changed", "define", "list alerts", "explain this metric"]
    complex_markers = [
        "binding constraint", "why didn't", "why did", "compare", "trade-off", "tradeoff",
        "sensitivity", "cvar", "stochastic", "progressive hedging", "alternative plan",
        "challenge", "red team", "what would invalidate", "least expensive", "within 5%",
        "scenario", "optimization", "monte carlo", "queue", "staffing", "transfer",
    ]
    if any(marker in text for marker in simple_markers):
        score -= 2; reasons.append("routine_status_or_summary")
    complex_hits = sum(marker in text for marker in complex_markers)
    if complex_hits:
        score += min(4, complex_hits); reasons.append("engineering_reasoning_markers")
    if tool_count >= 3:
        score += 2; reasons.append("multi_tool_investigation")
    elif tool_count >= 1:
        score += 1; reasons.append("tool_grounded_query")
    if requires_cross_module_reasoning:
        score += 2; reasons.append("cross_module_reasoning")
    if len(text) > 1200:
        score += 2; reasons.append("long_context")
    elif len(text) > 500:
        score += 1; reasons.append("moderate_context")
    if re.search(r"\b(why|how|which)\b", text) and complex_hits:
        score += 1; reasons.append("causal_or_comparative_explanation")

    sonnet_budget = _env_int("MINCO_CLAUDE_SONNET_DAILY_TOKEN_BUDGET", 250_000)
    budget_exhausted = sonnet_tokens_used_today >= sonnet_budget
    if score >= threshold and not budget_exhausted:
        return RoutingDecision(
            tier="deep", complexity_score=score, reason=tuple(reasons or ["complexity_threshold"]),
            max_output_tokens=_env_int("MINCO_CLAUDE_SONNET_MAX_OUTPUT_TOKENS", 2200),
            model_family="sonnet",
        )
    if budget_exhausted and score >= threshold:
        reasons.append("sonnet_daily_token_budget_exhausted")
    return RoutingDecision(
        tier="fast", complexity_score=score, reason=tuple(reasons or ["low_complexity"]),
        max_output_tokens=_env_int("MINCO_CLAUDE_HAIKU_MAX_OUTPUT_TOKENS", 900),
        model_family="haiku",
    )
