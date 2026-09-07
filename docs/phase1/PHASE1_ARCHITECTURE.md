# MINCO Phase 1 — Mathematical Hospital Engine

## Product identity

MINCO is a native healthcare-operations computational workstation in development. Phase 1 deliberately contains no new product UI. The mathematical system must remain technically interesting with every dashboard and LLM removed.

## Locked computational architecture

```text
Hospital event / replay evidence
        ↓
Poisson Hidden Markov / MMPP demand-regime inference
        ↓
Probabilistic demand + discharge hazard + ICU escalation ML
        ↓
Continuous-time Markov patient-flow model
        ↓
Queueing + conservation + staffed-capacity mathematics
        ↓
Monte Carlo stochastic futures (common random numbers)
        ↓
Two-stage stochastic MILP + CVaR
        ↓
Progressive Hedging / rolling-horizon reoptimization
        ↓
Human-reviewed action and evidence
```

## Architecture decision table

| Component | Phase-1 technology | Why MINCO uses it |
|---|---|---|
| Hidden demand state | Custom Poisson HMM / MMPP in NumPy/SciPy | Exposes the probabilistic mathematics instead of hiding it behind a generic classifier. |
| Patient flow | Continuous-time Markov chain | Hospital transitions occur in continuous time and resource time is itself an operational quantity. |
| Survival ML | Discrete-time hazard model | Predicts bed-release probability rather than treating length-of-stay as a plain regression target. |
| Escalation ML | Calibrated Gradient Boosting | Supplies a capacity-risk probability to stochastic scenario generation. |
| IE analytics | Erlang-C, M/M/s, Kingman, Little's Law, flow conservation, staffed-bed capacity | Healthcare flow and congestion are explicit engineering objects. |
| Uncertainty | Common-random-number Monte Carlo | Enables paired policy experiments on identical stochastic futures. |
| Primary OR | Julia + JuMP + Gurobi | Makes the stochastic MILP, CVaR, decomposition and rolling horizon a first-class mathematical engine. |
| Independent OR oracle | SciPy `milp` / HiGHS + exact enumeration | Solver correctness is not inferred from Gurobi alone. |
| LLM | Anthropic Claude with deterministic Haiku/Sonnet routing | Cost-aware evidence interrogation; never computes engineering results. |
| Phase-2 product UI | Flutter/Dart Windows workstation | Native hospital workstation rather than another browser dashboard. |
| Phase-2 interface | gRPC / Protocol Buffers | Strong typed boundary between Dart, Python and Julia runtimes. |

## Evidence boundary

Phase-1 numerical results are **VALIDATED ON SYNTHETIC GENERATIVE / BENCHMARK DATA**. Real-hospital calibration, clinical validation and production deployment are explicitly pending.
