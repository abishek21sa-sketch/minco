# MINCO Implementation Roadmap v1.0

This roadmap was authorized only after completion of the current-state audit. It preserves validated work and uses evidence gates rather than feature-count completion claims.

## Level 0 — Evidence Baseline

**Status:** Completed and accepted for execution.

Deliverables: archive inventory, current-state audit, evidence classifications, specification matrix, technical-debt register, risk register, preservation decisions, and stabilization actions.

Gate: every existing claim is linked to code, data, stored evidence, or explicitly classified as unverified.

## Level 1 — Stabilized Reproducible Core

**Purpose:** Make the preserved prototype safe to modify.

Work packages:

- L1-WP1 repository identity, packaging, and source hygiene;
- L1-WP2 locked dependency contract and solver preflight;
- L1-WP3 data-contract validation and provenance;
- L1-WP4 deterministic forecast and seeded simulation regression tests;
- L1-WP5 API packaging repair and solver-independent health endpoints;
- L1-WP6 smoke/standard/full execution profiles;
- L1-WP7 evidence manifest and claim qualification;
- L1-WP8 CI and containerized API baseline.

Acceptance gate:

- core tests pass in a clean environment;
- API `/health` works without Gurobi;
- solver endpoints fail transparently when the runtime is unavailable;
- the smoke pipeline completes in one command;
- repository-generated artifacts are governed and excluded from source control;
- all public claims use evidence-qualified language.

Specification alignment: Sections 1, 7, 15, 17, 23–25, 28, 31–39, 44, 48–50.

## Level 2 — Validated Analytical Core

**Purpose:** Strengthen the scientific and operational validity of the existing models.

Work packages:

- temporal and grouped forecasting validation;
- baseline and calibration governance;
- historical or benchmark calibration strategy;
- simulation verification and validation;
- formal stochastic/robust optimization formulation or corrected naming;
- optimization feasibility, sensitivity, and regression tests;
- uncertainty calibration and statistical claim governance;
- model/data registry with hashes, versions, and run manifests.

Gate: every analytical subsystem has baselines, validation methodology, failure tests, uncertainty reporting, and reproducible evidence.

Specification alignment: Sections 6, 8, 11–12, 17–22, 27, 31–33, 36, 41, 46, 49.

## Level 3 — Integrated Decision Intelligence Platform

**Purpose:** Convert research modules into an interoperable operational platform.

Work packages:

- ingestion adapters and canonical healthcare event model;
- operational-state repository and state-reconstruction service;
- scenario-management service;
- event-driven patient-flow simulation;
- optimization service contracts;
- decision-intelligence orchestration;
- grounded explainability and Copilot tool layer;
- dashboard migration from direct imports to API clients;
- role-specific workflows and audit context.

Gate: a versioned end-to-end workflow runs from validated operational events through recommendation, explanation, human review, and outcome logging.

Specification alignment: Sections 8–16, 19–27.

## Level 4 — Production Readiness

**Purpose:** Satisfy MINCO v1.0 engineering completion criteria.

Work packages:

- authentication and role-based authorization;
- privacy and security controls;
- data/model/solver observability;
- deployment environments and rollback;
- performance, resilience, and failure-mode testing;
- CI/CD quality gates;
- API versioning and backward-compatibility policy;
- user, operator, developer, testing, validation, and deployment documentation;
- release and open-source governance.

Gate: staging deployment passes security, resilience, reproducibility, observability, and documentation gates.

Specification alignment: Sections 23–40 and 44–45.

## Level 5 — Enterprise and Research Expansion

**Purpose:** Advance beyond MINCO v1.0 without destabilizing the validated core.

Candidates: multi-hospital tenancy, streaming integrations, causal intervention analysis, simulation-based optimization, federated learning, adaptive optimization, multi-agent simulation, and advanced grounded planning.

Gate: each extension demonstrates measurable operational value against Level 4 baselines and is introduced through versioned architecture decisions.

Specification alignment: Sections 41–50.
