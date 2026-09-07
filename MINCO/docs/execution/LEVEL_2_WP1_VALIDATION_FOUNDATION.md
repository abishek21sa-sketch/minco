# Level 2 Work Package 1 — Analytical Validation Foundation

**Release:** 0.3.0a1  
**Status:** Built; local Gurobi validation required before acceptance.

## Purpose

This work package establishes the evidence infrastructure needed to validate MINCO's forecasting, simulation, and optimization subsystems without relying on random row splits or untracked result files.

## Implemented

- real Gurobi package-and-license readiness preflight using a one-variable solve;
- immutable JSON run manifests with SHA-256 hashes for inputs, outputs, and the manifest itself;
- audit-database linkage from API what-if and event runs to their manifests;
- global chronological, grouped chronological, and leave-one-scenario-out split utilities;
- temporal and scenario-held-out forecast baseline evaluation;
- optimization feasibility, bound, elective-balance, transfer-arc, and repeatability checks;
- automated tests for each capability.

## Evidence boundary

This package creates validation infrastructure and baseline evidence. It does not approve the existing supervised models for production and does not convert the current state-transition simulator into a discrete-event digital twin.

## Acceptance commands

```powershell
pytest
python -m src.validation.forecast_baseline_validation
python -m uvicorn src.api.main:app --reload --port 8008
```

The `/readiness` response must report `solver_license_verified: true` on a machine with a valid Gurobi license. A new `/what-if` response must contain a manifest path and hash, and `/audit-log` must expose the same values.
