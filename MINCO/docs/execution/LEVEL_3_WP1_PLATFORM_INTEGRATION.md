# Level 3 WP1 — Operational State, Scenario, and Service Contracts

## Objective

Move validated analytical execution behind stable, versioned service contracts
without changing the validated simulation and optimization core.

## Delivered

- typed service contracts;
- operational-state reconstruction;
- governed built-in scenario catalog;
- audited scenario evaluation;
- decision recommendation orchestration;
- API dependency injection and `/v1` endpoints;
- backward-compatible legacy routes;
- contract registry with SHA-256 evidence;
- one-command integration gate.

## Acceptance gate

```powershell
pytest
python -m src.system.run_level3_wp1_validation
```

A passing gate requires Level 2 authorization, valid service schemas, reference
state reconstruction, scenario and decision execution, audit persistence,
manifest linkage, and explicit human review.

## Non-goals

This work package does not add real-time ingestion, a discrete-event twin,
authentication, distributed services, workflow queues, or real-hospital
calibration.
