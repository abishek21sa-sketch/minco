# MINCO Phase 1 Release Report — 0.6.0a3

## Disposition

**Candidate release:** Mathematical Hospital Engine — Julia dependency hotfix.

The previous `0.6.0a1` candidate is superseded by this dependency hotfix. `MathOptInterface` is now installed explicitly because `MINCOOptimization.jl` imports it directly. The previous `0.5.0a1` finalization candidate remains superseded. Phase 1 is not locked until the primary Julia/JuMP/Gurobi acceptance gate passes on the licensed Windows laptop.

## Build-environment validation

- Python test collection: 87 cases.
- Source validation executed in chunks because the packaging runner imposes a short command timeout: **84 passed, 3 skipped**.
- The three skips are historical Python Gurobi tests because `gurobipy` is absent from the packaging environment.
- New Phase-1 mathematical suite: **20 passed**.
- Mathematical validation gate: **PASSED**, 16/16 checks passed.
- Gate runtime: 14.38 seconds.

## Key synthetic validation evidence

- MMPP/HMM median regime-rate relative error: 0.168.
- Posterior regime recovery accuracy against synthetic hidden truth: 0.668.
- Discharge-hazard Brier improvement vs constant baseline: 8.15%.
- ICU-escalation ROC-AUC: 0.799.
- ICU-escalation Brier score: 0.141 vs constant 0.185.
- AI-parameterized stochastic MILP status: OPTIMAL.
- Independent feasibility maximum violation: 4.441e-15.
- Monte Carlo current-policy expected loss: 89.617.
- Monte Carlo surge-policy expected loss: 6.341.
- Monte Carlo current-policy CVaR95: 898.056.
- Monte Carlo surge-policy CVaR95: 126.152.
- HiGHS extensive-form objective matches exact enumeration within 0.000e+00.
- Progressive Hedging reference converged: True.

All results above are **synthetic generative / benchmark evidence**. They are not real-hospital performance claims.

## Primary OR runtime external gate

The primary optimizer is Julia/JuMP/Gurobi. Julia and Gurobi are not installed in the packaging environment, so Phase 1 requires laptop acceptance:

```powershell
julia julia/setup.jl
python -m src.system.run_phase1_primary_or_acceptance
```

This compares the Julia/Gurobi extensive-form objective against the independent Python/HiGHS oracle and runs the Julia Progressive Hedging acceptance script.

## Claude

Anthropic Claude is optional in Phase 1. The deterministic router selects Haiku-family or Sonnet-family work based on query complexity, tool count, cross-module reasoning, routing mode and a configurable Sonnet daily token budget. No API key is required to validate the mathematical engine.

## Evidence manifest

- Validation manifest SHA-256: `29f9605daf1eea2d6f84bdc0fb3416be57fe61f6c38e3bb92b4c699c2f4961f4`

## Julia acceptance hotfix a3

The licensed Windows acceptance run validated the primary Julia/JuMP/Gurobi extensive-form stochastic MILP as `OPTIMAL`, objective `12.86`, relative gap `0.0`, and exact objective agreement with the independent Python/HiGHS oracle. The remaining Progressive Hedging script then failed before completion because `dot` was referenced without importing `LinearAlgebra`. Version `0.6.0a3` removes that namespace dependency and adds a regression check. Full Progressive Hedging runtime acceptance remains laptop-gated until rerun.
