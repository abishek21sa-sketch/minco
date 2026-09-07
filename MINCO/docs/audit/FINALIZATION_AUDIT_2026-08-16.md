# MINCO Finalization Audit — 2026-08-16

## Audit basis

The latest available release material was a Level 3 WP1 patch rather than a standalone repository. The effective candidate baseline was reconstructed from the last full stabilized repository plus every accepted patch through Level 3 WP1, then executed and inspected as one repository. The governing MINCO Engineering Specification v1.0 remains in force.

## Current genuine completeness

**60% of the approved V1.0 target, medium confidence.**

This percentage is not inherited from earlier release targets. It reflects implementation evidence against the full specification. The analytical research core is strong, but material V1.0 product requirements remain: true event/state ingestion, a patient-flow DES, a distinctive production UI, external-data replay, observability/security hardening, final technical documentation, and exact-release acceptance.

## What is already strong

- **IMPLEMENTED / TESTED:** typed FastAPI/Pydantic service boundary, scenario service, decision service, audit persistence, immutable run manifests, data validation, Markov state-transition model, seeded stochastic simulation, Gurobi network capacity optimization, scenario/sensitivity tooling.
- **VALIDATED ON SYNTHETIC BENCHMARK:** Markov expectation convergence, mass conservation, arrival-process calibration, solver feasibility/repeatability, sensitivity monotonicity, hash-addressed evidence.
- **PRESERVE:** service contracts, loader/data contracts, Markov transition engine, stochastic state-transition simulator, Gurobi model structure, audit database, run-manifest system, scenario catalog, evidence language and claim boundaries.

## What is broken or weak

1. **BROKEN decision semantics — transfer capacity.** `allowed` was a binary topology flag but the what-if layer could multiply it as though it were capacity. A 0.8 multiplier therefore made `allowed=0.8`, which removed the arc from `allowed == 1` filtering. Transfer degradation scenarios were not modeling partial capacity.
2. **UNVERIFIED operational AI claim.** The prior WP3 forecasting dataset assigned `time_index` across independent Monte Carlo replications and formed next-step targets by shifting replication rows. It is actual supervised ML, but it is not a defensible operational time-series forecast. That approval is superseded for finalization.
3. **PARTIALLY IMPLEMENTED AI-to-decision coupling.** Earlier recommendation logic was threshold/rule driven and did not consume a defensibly validated demand forecast.
4. **PARTIALLY IMPLEMENTED IE mathematics.** Markov resource demand was coded, but explicit recruiter-readable capacity equations such as Little's Law were not a first-class tested decision layer.
5. **PARTIALLY IMPLEMENTED OR reporting.** The optimization is real, but public API responses historically emphasized KPIs rather than the optimized surge, elective, and transfer actions.
6. **PARTIALLY IMPLEMENTED digital twin.** Current simulation is a seeded stochastic cohort state-transition twin, not a patient-level discrete-event or continuously synchronized hospital twin.
7. **WEAK product UX.** The existing Streamlit interface is a large dashboard module and does not provide a distinctive hospital surge-response workflow.
8. **MISSING V1 product controls.** Real/batch event ingestion, state freshness, final observability/security posture, public release hygiene, and final technical methods documentation remain incomplete.

## What is missing for genuine V1.0

- canonical operational event/batch ingestion and replay;
- persisted/fresh operational-state reconstruction;
- patient-flow discrete-event simulation with queue/wait/throughput measures;
- project-specific Surge Response Workbench frontend;
- evidence-grounded Gemini explanation/challenge layer if deterministic workflows are ready;
- performance/caching and observability appropriate to solver/simulation workloads;
- final security boundary, dependency cleanup, warning cleanup, secret scan, license selection;
- `docs/TECHNICAL_METHODS.md`;
- exact final ZIP clean-extraction validation and short Windows/Gurobi acceptance;
- GitHub/public V1.0 cleanup.

## Actual AI currently present

- Historical supervised models and regime experiments exist, but their earlier replication-index temporal interpretation is downgraded to **RESEARCH / SYNTHETIC EVIDENCE**.
- Finalization Phase 1 introduces a **real time-axis next-day quantile demand forecast** trained on generated synthetic historical replay, with chronological train/calibration/test partitions, seasonal-naive baseline, and conformalized uncertainty.

## Actual Industrial Engineering currently present

- patient-flow state-transition modeling;
- resource/capacity utilization logic;
- safe-capacity thresholds;
- surge capacity and elective flow decisions;
- Finalization Phase 1 adds an absorbing-Markov fundamental-matrix calculation and **Little's Law** translation from arrivals/day and expected resource-days/arrival to expected census and capacity pressure.

## Actual Operations Research currently present

A genuine Gurobi network capacity model with decision variables for surge capacity, elective admissions/rejections, ICU transfer load, unsafe slack, and overflow slack; objective penalties/costs; capacity, elective-balance, transfer-flow, and surge constraints. Finalization Phase 1 adds explicit per-lane transfer-capacity constraints and solver action summaries.

## Proposed distinctive technical identity

**Signature capability: Hospital Surge Response Workbench.**

Forecast P05/P50/P95 next-day arrivals → translate demand into hospital/resource pressure → optimize surge/elective/transfer actions → stochastic stress-test the plan → present a human-reviewed operational action with evidence and uncertainty.

## Architecture decision table

| Component | Chosen technology | Why it fits MINCO | Why not the default elsewhere |
|---|---|---|---|
| Backend/API | FastAPI + Pydantic services | Existing typed contracts are good and analytically oriented; preserve rather than rewrite | Diversity should not destroy a validated service layer |
| Frontend target | SvelteKit Surge Response Workbench | Well suited to a fast interactive capacity grid, scenario timeline, and decision drawer | Replaces the generic Streamlit dashboard metaphor rather than copying another portfolio dashboard |
| Audit store | SQLite | Local deterministic decision/audit trail is already reliable and portable | No need for a server DB just for portfolio scale |
| Historical/event analytical store | DuckDB + Parquet (Phase 2) | Fits local hospital-event replay and time-series analytical queries | Adds a justified columnar layer rather than technology bingo |
| AI/ML | Scikit-learn quantile Gradient Boosting + conformal intervals | Forecast uncertainty feeds planning decisions directly | Healthcare needs calibrated demand uncertainty, not the portfolio's other AI patterns |
| IE | Absorbing Markov flow math + Little's Law + queue/capacity measures | Directly translates patient-flow uncertainty into resource consequences | Healthcare-specific flow/capacity reasoning |
| OR/Solver | Gurobi network capacity optimization | Transfers, surge, elective control, and future staffing/bed decisions are constrained network decisions | Gurobi is used because the decision model benefits from it, not for branding |
| Simulation | Current seeded stochastic twin; patient-flow DES in Phase 2 | Monte Carlo remains useful for uncertainty; DES is needed for wait/queue/throughput validation | Healthcare patient flow requires event timing rather than a manufacturing animation |
| LLM | Gemini, optional and tool-grounded | Explain/challenge deterministic evidence, assumptions, and scenarios | LLM will sit above engineering models, not invent numbers |
| Deployment | Local/Docker-first | Gurobi licensing and healthcare-style data boundaries make local decision-workbench deployment defensible | A public URL is not worth weakening solver/security boundaries |

## Minimum remaining phases

**Three.**

1. **Phase 1 — Decision Engine Closure & Mathematical Correctness.** Repair transfer capacity, replace invalid temporal AI semantics, add tested IE math, expose optimized actions, and couple AI → IE → OR → simulation.
2. **Phase 2 — Operational Replay, DES & Distinctive Product.** Canonical event ingestion/state freshness, DuckDB/Parquet replay, patient-flow DES, SvelteKit Surge Response Workbench, Gemini evidence explanation if warranted.
3. **Phase 3 — V1.0 Hardening & Public Release.** Security/observability/dependency cleanup, technical methods, README, benchmarks, exact clean-extract release, secret scan, 1.0.0 normalization, GitHub release and closure.

## What still requires the user's laptop / credentials / real-world validation

- Gurobi-licensed execution of solver-backed tests and integrated surge-plan gate;
- Windows/browser acceptance of the exact distributed ZIP;
- Gemini key only in Phase 2 if/when the grounded explanation layer is enabled (`.env`, never chat);
- real-hospital or credible historical operational data for external predictive/simulation validation. Until then all predictive and outcome claims remain synthetic-reference-case evidence.
