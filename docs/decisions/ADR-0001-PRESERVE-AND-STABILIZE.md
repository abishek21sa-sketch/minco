# ADR-0001: Preserve and Stabilize the Existing MINCO Core

**Status:** Accepted  
**Date:** 2026-07-27

## Decision

Preserve the existing forecasting, stochastic simulation, optimization, scenario analysis, explanation, command-center, and audit concepts. Stabilize them before introducing new platform capabilities.

## Rationale

The current-state audit found meaningful implemented and validated work. Restarting would discard evidence and violate the governing specification principle to preserve validated work unless replacement is justified.

## Consequences

- Existing numerical claims remain tied to archived evidence.
- New features must pass tests and evidence gates.
- Terminology must match implementation fidelity.
- Large generated models and bytecode are removed from the source release but recorded in an artifact manifest.
