from src.claude.router import route_claude_query


def test_claude_router_uses_fast_tier_for_routine_status_and_deep_for_decision_review(monkeypatch):
    monkeypatch.setenv("MINCO_CLAUDE_ROUTING_MODE", "balanced")
    simple = route_claude_query("Summarize current status and list alerts")
    deep = route_claude_query(
        "Compare the stochastic optimization scenarios, explain the binding constraint, "
        "challenge the CVaR plan, and identify the least expensive alternative within 5% of service level.",
        tool_count=5,
        requires_cross_module_reasoning=True,
    )
    assert simple.model_family == "haiku"
    assert deep.model_family == "sonnet"
    assert deep.complexity_score > simple.complexity_score


def test_claude_router_downgrades_when_sonnet_budget_is_exhausted(monkeypatch):
    monkeypatch.setenv("MINCO_CLAUDE_SONNET_DAILY_TOKEN_BUDGET", "100")
    decision = route_claude_query(
        "Compare optimization and Monte Carlo sensitivity and explain binding constraints",
        tool_count=4,
        requires_cross_module_reasoning=True,
        sonnet_tokens_used_today=100,
    )
    assert decision.model_family == "haiku"
    assert "sonnet_daily_token_budget_exhausted" in decision.reason
