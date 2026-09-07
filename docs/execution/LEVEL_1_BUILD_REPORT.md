# MINCO Level 1 Build Report

Release: `0.2.0a1`  
Engineering level: Stabilized Reproducible Core

## Implemented in this build

- canonical MINCO product identity with Meridian retained only as the synthetic case;
- package/dependency contract using `pyproject.toml`;
- optional solver and dashboard dependencies;
- stronger CSV uniqueness, referential-integrity, bounds, and transition validation;
- repaired FastAPI entry point at `src.api.main:app`;
- lazy solver imports so `/health` and `/readiness` work without Gurobi;
- automated loader, Markov, simulation, evidence-claim, API, and optional solver tests;
- fast solver-independent smoke pipeline;
- repository hygiene rules and external model-artifact manifest;
- CI baseline;
- API container baseline;
- data dictionary, claim register, architecture note, ADR, and evidence-gated roadmap.

## Explicitly not claimed

This build does not complete MINCO v1.0. It does not introduce real hospital data, DES, live ADT reconstruction, formal robust optimization, production security, full observability, or a general grounded Copilot.
