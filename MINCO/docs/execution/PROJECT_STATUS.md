# MINCO Project Status

**Mode:** Enterprise product hardening  
**Current release:** 0.5.0a1  
**Current phase:** Phase 3 — Enterprise Security, Observability & Release Hardening  
**Repository disposition:** Preserve, repair, finish  
**Genuine V1.0 completeness from finalization audit:** 60% (medium confidence)  

## Enterprise hardening milestone delivered

- enterprise control-tower read model composed from the existing operational-state service and FLOW-CVaR signature algorithm;
- versioned `GET /v1/control-tower` API contract with network, patient-flow, risk, decision, diagnostics, evidence, scenario, and governance domains;
- Streamlit Enterprise Control Tower surface for operator and executive review;
- explicit separation of mathematical `AUTHORIZED`, human `HUMAN_REVIEW`, and failed `BLOCKED` states;
- artifact hashes, input fingerprints, status freshness, solver preflight, model scope, and synthetic-only claim boundary surfaced together;
- append-only human-review workflow with reviewer identity, governed disposition, rationale, and audit retrieval endpoints;
- correlation-aware structured API telemetry and typed liveness/readiness contracts with explicit non-authorization semantics;
- Julia/JuMP/Gurobi dependency preflight that prevents a present-but-uninstantiated Julia executable from breaking the Phase 2 gate;
- Phase 2 backend gate evidence: 459 canonical replay events, DuckDB/Parquet/Polars round-trip passed, optimal workstation plan, and explicit SciPy/HiGHS fallback while Julia dependencies remain uninstantiated;
- enterprise operational-history endpoint with DuckDB/Parquet-or-memory fallback, reconstructed replay state, event-type provenance, and explicit replay-only freshness semantics;
- governed external-event ingestion boundary with typed status/ingest contracts, source provenance, freshness windows, idempotent event IDs, and raw patient-identifier rejection;
- tamper-evident human-review audit chain with SHA-256 predecessor hashes, legacy-schema migration, and a verification endpoint surfaced in the Control Tower;
- low-cardinality SLO metrics with typed JSON and Prometheus outputs, normalized route labels, and explicit no-payload/no-identifier retention policy;
- hash-addressed `GET /v1/release-evidence` attestation over the contract registry, backend gate, runtime evidence, manifest, and status record, with an explicit integrity-only claim boundary;
- reproducible `scripts/generate_release_evidence.py` handoff command producing `results/validation/release_evidence_attestation.json`;
- repeatable `RUN_ENTERPRISE_ACCEPTANCE.cmd` candidate-package harness with a documented live API, API-key, ingestion, and evidence-review procedure;
- governed `RUN_APP.cmd` launcher now serves the versioned API and executive Command Center on port 8814; the legacy deterministic demo is explicitly separated under `RUN_DEMO.cmd`;
- machine-readable `GET /v1/release-readiness` scorecard that joins artifact evidence, solver/dependency posture, security controls, blockers, and next actions;
- Operations Workbench with governed baseline/counterfactual scenario comparison, metric deltas, selectable replication count, reviewer disposition, and immutable review-history refresh;
- portable decision-packet endpoint bundling run provenance, manifests, reviews, audit integrity, release evidence, blockers, and an explicit non-autonomous authorization boundary;
- enterprise identity posture with redacted `/v1/identity`, API-key/Bearer support, configurable role hierarchy, and reviewer-gated disposition writes;
- optional deployment-boundary API-key enforcement for operational routes, with public liveness/readiness probes and a scorecard check that distinguishes configured credentials from actual enforcement;
- CI workflow now runs the Phase 2 backend gate and validates the release-readiness contract without exporting evidence artifacts;
- complete Python regression suite: 135 passed after the Operations Workbench, decision-packet, and enterprise identity/RBAC increments.

This milestone improves product coherence and decision governance without changing the validated FLOW-CVaR mathematical core.

## Phase 1 implemented in the candidate build

- genuine date-axis synthetic historical replay;
- probabilistic next-day arrival AI with chronological validation, seasonal-naive baseline, and conformalized intervals;
- absorbing-Markov expected resource-days and Little's Law capacity translation;
- explicit transfer-lane capacity separate from binary topology;
- Gurobi lane-capacity constraints;
- optimized action reporting;
- integrated AI → IE → OR → stochastic simulation surge-plan service;
- updated evidence/claim governance;
- finalization audit and three-phase closure plan.

## Validation state

The current Windows build passes the repository test suite with only three licensed-solver checks skipped in this environment. The Phase 2 backend gate and product runtime acceptance are green; the release-evidence attestation verifies the candidate package contents.

## Remaining enterprise gates

- Approved ADT/EHR or event-stream connection with data-contract validation and privacy review.
- Enterprise identity/access control, secret rotation, TLS/transport policy, and deployment-platform integration.
- Julia/JuMP/Gurobi/JSON3 environment instantiation and licensed-solver verification where required by deployment.
- External hospital validation, performance/SLO burn-in, disaster-recovery evidence, and formal clinical/operational governance review.

## Evidence boundary

All current prediction, simulation, and optimization performance evidence is synthetic. Real-hospital and clinical validation remain external-validation work and are not prerequisites for an honest portfolio V1.0 if labeled explicitly.
