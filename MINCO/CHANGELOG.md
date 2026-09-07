# MINCO Changelog

## 0.8.0a4 — Julia JSON3 registry UUID hotfix

- Correct the direct dependency UUID for JSON3 to the registered General-registry UUID `0f8b85d8-7281-11e9-16c2-39a750bddbf1`.
- Retain `Pkg.resolve() -> Pkg.instantiate() -> Pkg.precompile()` so merge-in upgrades reconcile stale manifests before runtime.
- Add a regression assertion for the exact JSON3 UUID, preventing future package-name-only checks from accepting an invalid Julia project.

## 0.8.0a3

- Fix Julia merge-upgrade setup by resolving stale Manifest.toml files before instantiation.
- Add regression coverage requiring `Pkg.resolve()` before `Pkg.instantiate()`.
- Preserve explicit direct Julia dependencies introduced in 0.8.0a2.

## 0.8.0a2 — Julia environment packaging hotfix

- Declared JuMP, MathOptInterface, Gurobi, HiGHS, and JSON3 as direct Julia project dependencies instead of relying on a mutable setup-time add step.
- Changed `julia/setup.jl` to reproducibly `Pkg.instantiate()` and precompile the declared project.
- Added a regression contract test that fails when the distributed Julia `Project.toml` omits any runtime dependency.
- Fixes the Phase-2 workstation error `Package JuMP not found in current path` after a clean repository extraction.

## 0.8.0a1 — Native hospital operations workstation

- Added canonical hospital event model with explicit provenance and validation.
- Added DuckDB operational event lake, ordered Parquet export and Polars replay analytics.
- Added deterministic synthetic historical replay plus state reconstruction and freshness semantics.
- Added patient-level discrete-event simulation with conservation checks and intervention comparison.
- Added gRPC + Protocol Buffers runtime boundary for the native client; REST is not the primary Phase-2 product path.
- Added Flutter/Dart Windows workstation source with Census Board, Patient Flow Theatre, Monte Carlo Room, Intervention Composer and Decision Review Board.
- Wired workstation planning to the stochastic OR engine, including the Julia/JuMP/Gurobi primary path on licensed machines.
- Wired Decision Review to deterministic evidence and the adaptive Anthropic Claude Haiku/Sonnet router.
- Added Phase-2 data/runtime/workstation regression tests and Windows bootstrap/start scripts.

## 0.6.0a3 — Julia Progressive Hedging namespace hotfix

- Fixed Windows Julia acceptance failure `UndefVarError: dot not defined` in `progressive_hedging.jl`.
- Replaced the unnecessary `LinearAlgebra.dot` dependency with explicit probability-weighted summation: `sum(probabilities .* objectives)`.
- Added a regression assertion to the Julia engine contract test.
- Preserves the already validated JuMP/Gurobi extensive-form result (objective 12.86, zero MIP gap, exact agreement with the independent HiGHS oracle).

# Changelog

## 0.5.0a1 — Finalization Phase 1

### Added

- genuine calendar-time synthetic arrival history for operational forecast validation;
- chronological quantile Gradient Boosting P05/P50/P95 arrival forecasting with a seasonal-naive baseline and conformal interval calibration;
- executable absorbing-Markov expected resource-days and Little's Law hospital-capacity translation;
- integrated `POST /v1/surge-plan` AI → IE → OR → stochastic-simulation workflow;
- optimized action summaries and solver-status reporting;
- explicit `transfer_capacity` data parameter and Gurobi per-lane capacity constraints;
- finalization audit, three-phase closure plan, and ADRs for forecast-time semantics and transfer topology/capacity;
- `/finalization-status` endpoint and current operational model-governance evidence.

### Corrected

- transfer-capacity scenarios no longer scale the binary `allowed` topology flag;
- prior replication-index prediction experiments are no longer treated as operational temporal forecast validation;
- request `planning_quantile` now actually controls the surge-plan demand guardrail.

### Evidence boundary

- current AI/IE validation is synthetic-reference-case evidence;
- historical optimization outcome claims generated before the transfer-capacity correction are legacy evidence pending a corrected-model rerun.

## 0.4.0a1 — Level 3 WP1

### Added

- versioned Pydantic contracts for operational state, scenarios, metrics, and recommendations;
- operational-state reconstruction service with source fingerprints;
- built-in scenario catalog and audited scenario-evaluation service;
- decision-orchestration service with explicit human-review recommendations;
- application service composition root and dependency injection for API tests;
- `/v1/operational-state`, `/v1/scenarios`, `/v1/scenarios/evaluate`, and `/v1/decisions/recommend`;
- hash-addressed service-contract registry and Level 3 WP1 integration gate;
- integration, contract, persistence, and backward-compatibility tests.

### Changed

- legacy `/what-if`, `/event`, and `/recommend` routes now delegate to application services;
- API transport code no longer performs direct analytical orchestration;
- version advanced to `0.4.0a1`.


## 0.3.0a1 — 2026-07-29

### Added

- Gurobi package-and-license readiness verification through an actual preflight solve;
- SHA-256 run manifests for API what-if, event, and forecast-validation executions;
- audit linkage from operational runs to immutable manifest evidence;
- chronological, grouped chronological, and leave-one-scenario-out validation splits;
- forecast persistence and majority-class baselines under time-respecting validation;
- optimizer feasibility, repeatability, bound, elective-balance, and transfer-arc checks;
- Level 2 WP1 validation documentation and tests.

### Changed

- `/readiness` now distinguishes ready, degraded, and not-ready states using verified solver evidence;
- `/audit-log` now includes manifest paths and hashes;
- version advanced to `0.3.0a1`.


## 0.2.0a1 — 2026-07-27

### Added

- evidence-gated implementation roadmap;
- stronger data-contract validation;
- automated regression tests;
- canonical metrics subsystem;
- solver-independent smoke pipeline;
- repaired FastAPI service entry point and readiness checks;
- environment, evidence, and artifact manifests;
- CI and container baselines;
- current architecture, data dictionary, claims register, ADR, and Level 1 build report.

### Changed

- established MINCO as the product/repository identity;
- retained Meridian Health Network as the synthetic reference case only;
- renamed the UI's digital-twin view to stochastic state-transition twin;
- qualified robust-policy and scalability claims according to available evidence.

### Removed from source release

- compiled Python bytecode;
- 58 large pickle model artifacts, preserved by hash in `archive/manifests/EXTERNAL_MODEL_ARTIFACTS.csv`;
- empty placeholder notebooks and unused one-off editing scripts.

## 0.3.0a2 — 2026-07-30

### Added

- Monte Carlo verification of patient-mass conservation and convergence to deterministic Markov expectations;
- synthetic Poisson arrival-process calibration checks;
- paired bootstrap claim-governance classifications separating numerical effects from inferential support;
- capacity, demand, and transfer monotonicity checks for the optimization model;
- canonical uncertainty-adjusted optimization naming and backward-compatible legacy metadata;
- Level 2 WP2 one-command validation and evidence manifests.

### Changed

- advanced the analytical validation release to `0.3.0a2`;
- explicitly classified the legacy `robust_optimized_network` method as arrival-stress-adjusted nominal optimization, not formal robust optimization.

## 0.3.0a3 — Level 2 WP3

- Added chronological and scenario-held-out predictive-model governance.
- Added explicit future-feature and deterministic-target dependency audits.
- Added baseline-relative approval, conditional, and rejection decisions.
- Added classification calibration metrics.
- Added portable model registry entries with dataset and artifact SHA-256 hashes.
- Added one-command WP3 validation and automated governance tests.


## 0.3.0a4 — Level 2 WP4

### Added

- synthetic benchmark calibration-readiness contract and external-validity boundary;
- hash-addressed data, model, and validation asset registry;
- recursive run-manifest integrity verification;
- one-command Level 2 completion gate;
- `/validation-status` and `/evidence-catalog` API endpoints;
- explicit authorization boundary between Level 2 analytical validation and Level 3 platform integration.

## 0.6.0a1 — Mathematical Hospital Engine candidate

- Supersedes the conservative `0.5.0a1` finalization candidate architecture.
- Adds a custom Poisson Hidden Markov / MMPP demand-regime engine with Baum-Welch EM.
- Adds continuous-time Markov patient-flow mathematics and population simulation.
- Adds Erlang-C M/M/s, Kingman, Little's Law, conservation and staffed-capacity IE calculations.
- Adds discrete-time discharge-hazard ML and calibrated ICU-escalation Gradient Boosting.
- Adds AI-parameterized stochastic ICU-census scenario generation.
- Adds common-random-number Monte Carlo policy experiments and CVaR evidence.
- Adds Julia/JuMP/Gurobi two-stage stochastic MILP, Progressive Hedging and rolling-horizon source runtime.
- Adds independent SciPy/HiGHS MILP, exact enumeration and feasibility oracles.
- Adds deterministic cost-aware Anthropic Claude Haiku/Sonnet routing foundation.
- Locks Phase 2 to a native Flutter/Dart Windows hospital workstation with gRPC/Protocol Buffers.

## 0.6.0a2 — Julia dependency hotfix

- Adds `MathOptInterface` as an explicit Julia setup dependency because the MINCO Julia module imports it directly.
- Extends the Julia dependency contract regression test to prevent recurrence.
- Improves the Python→Julia bridge error with a targeted setup remediation when a Julia package is missing.
- No mathematical formulation or benchmark logic changed from `0.6.0a1`.
