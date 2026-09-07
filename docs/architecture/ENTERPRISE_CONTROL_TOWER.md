# Enterprise Control Tower

## Purpose

The enterprise control tower is MINCO's operator-facing read model. It joins the current reference-network state, Markov patient-flow forecast, FLOW-CVaR recommendation, tail-risk counterfactual, solver diagnostics, scenario catalog, evidence provenance, and authorization state in one versioned service response.

It is an aggregation boundary over existing MINCO services. It does not replace the FLOW-CVaR optimizer, introduce a weighted recommendation score, or permit autonomous hospital actions.

## Service contract

The canonical endpoint is `GET /v1/control-tower`. The response has `schema_version = 1.0` and exposes these domains:

- `network`: hospital inventory, capacity totals, input fingerprints, projected census, and explicit model scope;
- `patient_flow`: states, row-stochastic transition matrix, propagated state, and predicted next state;
- `risk`: expected recourse, tail-scenario loss, CVaR parameters, scenario probabilities, and the No-CVaR counterfactual;
- `decision`: FLOW-CVaR decision ID, first-stage actions, scenario recourse, and bounded claim;
- `diagnostics`: reference solver, production solver preflight, model size, and feasibility checks;
- `evidence`: evidence-class separation, artifact hashes, data freshness, and claim boundary;
- `operational_history`: canonical event count, event-type distribution, reconstructed replay state, data-stack provenance, and freshness posture;
- `governance`: human-review state, failed checks, release readiness, explicit autonomous-execution prohibition, and the durable review workflow contract.

## Governance semantics

`AUTHORIZED` in the signature payload means that the mathematical/evidence checks passed for the supplied reference case. The control tower separately reports `HUMAN_REVIEW` because the result is not an operational authorization. A missing or failed formulation/evidence check produces `BLOCKED`.

The current release reports `NOT_FOR_PRODUCTION` because the bundled data is synthetic, the operational status artifact is not a live ADT/EHR feed, and real hospital validation is not present. This is intentional claim discipline, not a UI-only status label.

The canonical event-history endpoint is `GET /v1/operational-history`. It prefers the embedded DuckDB event lake with ordered ZSTD Parquet export and falls back to the deterministic in-memory replay adapter when optional dependencies are unavailable. Both paths expose the same contract and mark the current bundled events as `REPLAY_ONLY`; a replay timestamp is not a live-data freshness assertion.

The external-event boundary is exposed through `GET /v1/operational-events/status` and `POST /v1/operational-events/ingest`. The gateway accepts bounded `external_stream` batches only after canonical schema validation, source-system matching, timezone/freshness checks, event-id deduplication, and raw patient-identifier rejection. It appends to the same event store used by history reconstruction, but its availability does not mean an ADT/EHR feed is connected; the reference package remains explicitly disconnected until an approved source is configured.

The release-governance endpoint is `GET /v1/release-readiness`. It composes the contract registry, product-runtime evidence, Phase 2 backend evidence, solver preflight, and deployment security posture into a typed scorecard. The scorecard is intentionally `NOT_FOR_PRODUCTION` for the bundled reference package: it names missing live ADT/EHR connectivity, enterprise identity/access control, external validation, and solver licensing as blockers instead of converting engineering evidence into a production claim.

## Human review workflow

Every `/v1/decisions/recommend` response produces a persisted `run_id`. An authorized operator can append a human disposition through `POST /v1/audit/{run_id}/review` using one of three governed outcomes: `ACCEPTED_FOR_OPERATIONS_REVIEW`, `DEFERRED`, or `REJECTED`. The event captures the reviewer, UTC timestamp, decision, and comment in the SQLite audit database.

Review events are append-only and retrievable through `GET /v1/audit/{run_id}/reviews` or `GET /v1/audit/reviews`. Each event carries a SHA-256 hash, its predecessor hash, and the `GENESIS` anchor; `GET /v1/audit/integrity` verifies the chain and reports violations without repairing non-null records. The contract hard-codes `autonomous_execution_permitted = false`; even the positive disposition means “accepted for operations review,” not approved for execution. The control tower exposes the endpoint, allowed dispositions, chain-integrity status, immutable-log status, and the same prohibition so the operating procedure is visible at the point of review.

## Enterprise operating standard

The surface is designed around an operating decision cycle:

1. Establish the network picture and data freshness.
2. Inspect predicted flow and where the model is in scope.
3. Review expected and tail risk together.
4. Inspect the actual integer capacity and recourse actions.
5. Verify solver status, evidence hashes, and gate checks.
6. Reconcile against current hospital facts and append a human disposition with reviewer identity and rationale.
7. Treat the review event as governance evidence only; any operational action remains outside MINCO and requires separate local authorization.
8. Later compare predicted versus realized operations when an approved data source exists.

The present implementation completes steps 1 through 7 for the synthetic reference case and provides the governed adapter boundary for step 8. Live ADT/EHR connectivity, identity and access control, multi-hospital FLOW-CVaR expansion, and realized-outcome logging remain separate release work because they require external operational authority and data.

## Operational telemetry and readiness

The API boundary now returns typed `HealthResponse` and `ReadinessResponse` contracts. `GET /health` is a liveness check that does not require Gurobi. `GET /readiness` reports data-asset availability, solver package/license state, a check map, and the explicit `NOT_FOR_PRODUCTION`/no-autonomous-execution boundary.

Each request receives a validated `X-Correlation-ID` response header. A caller-supplied safe ID is preserved; malformed or oversized values are replaced with a generated ID. The `minco.api` logger emits structured completion events containing method, path, status, duration, and correlation ID, while deliberately excluding query strings and request payloads.

When `MINCO_API_KEY` or `MINCO_API_KEYS` is present in the deployment environment, operational routes require constant-time credential validation through `X-API-Key` or `Authorization: Bearer`. `MINCO_API_KEYS` accepts a JSON mapping from credential to redacted identity metadata, for example `{"role":"reviewer","subject":"operations-reviewer"}`. The effective role hierarchy is `viewer < analyst < reviewer < admin`; evaluation and ingestion require `analyst`, while human disposition writes require `reviewer`. `GET /v1/identity` exposes the redacted subject, role, permissions, and enforcement posture without returning credentials. `/health`, `/readiness`, documentation routes, and CORS preflight remain available to platform probes. An unset key intentionally leaves the local reference package in `LOCAL_REFERENCE_CASE_NO_AUTH` mode; the release scorecard reports that mode as a blocker.

The observability boundary exposes typed metrics at `GET /v1/observability/metrics` and Prometheus text at `/metrics`. Counters are normalized to low-cardinality routes and include request totals, 5xx errors, status classes, and latency summaries. Metrics intentionally exclude query strings, request payloads, patient identifiers, and correlation IDs.

The release-readiness scorecard is the deployment-review handoff: it is suitable for CI/CD policy checks and executive review, but it does not grant operational authorization. Human review remains required and autonomous execution remains prohibited.
