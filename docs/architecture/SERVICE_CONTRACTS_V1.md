# MINCO Service Contracts v1

## Purpose

Level 3 introduces a stable application-service boundary between API transports
and analytical implementations. API routes validate and serialize requests;
services orchestrate state reconstruction, simulation, optimization, audit, and
evidence generation.

## Contract version

Public service contracts use `contract_version = "1.0"`. Additive fields may be
introduced within v1. Breaking field or semantic changes require a new contract
version and new `/v2` endpoints.

## Services

### Operational State Service

Reconstructs the current synthetic reference state from validated input tables.
It reports hospitals, cohorts, horizon, expected arrivals, capacities, safe
thresholds, transfer connectivity, and input fingerprints. It does not claim to
be a live ADT or EHR feed.

### Scenario Service

Owns the scenario catalog, parameter validation, analytical execution, audit
persistence, and immutable run manifests. The service delegates to the existing
stochastic state-transition simulator and continuous network LP.

### Decision Orchestration Service

Transforms scenario metrics into alert classifications and human-review
recommendations. It persists decision, recommendation, alert, and manifest
evidence. It does not autonomously implement operational actions.

## Transport boundary

Versioned endpoints:

- `GET /v1/operational-state`
- `GET /v1/scenarios`
- `GET /v1/scenarios/{scenario_id}`
- `POST /v1/scenarios/evaluate`
- `POST /v1/decisions/recommend`

Legacy routes remain compatibility aliases and delegate to these services.

## Claim boundary

All responses remain restricted to the bundled synthetic Meridian reference
case. No contract implies real-hospital calibration, clinical validity, or
approval for autonomous patient-care decisions.
