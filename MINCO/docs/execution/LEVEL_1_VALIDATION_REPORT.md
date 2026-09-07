# MINCO Level 1 Validation Report

**Release:** 0.2.0a1  
**Validation date:** 2026-07-27  
**Scope:** Stabilized Reproducible Core

## Results

- Python source compilation: passed.
- Automated tests: .................s.                                                      [100%].
- Solver-independent smoke pipeline: passed.
- Base synthetic instance: loaded and validated.
- 10, 25, 50, 100, and 200-hospital synthetic instances: loaded and validated.
- Deterministic Markov forecast repeatability: passed.
- Expected patient-mass conservation: passed.
- Seeded stochastic simulation reproducibility: passed.
- Headline result calculations against stored CSV evidence: passed.
- FastAPI `/health`, `/readiness`, and empty audit-log behavior: passed.
- Solver-backed endpoint behavior without Gurobi: returned explicit HTTP 503 as designed.
- Editable package installation: passed using the existing environment with build isolation disabled.
- Independent licensed Gurobi rerun: not executed in this environment.

## Artifact governance

Fifty-eight pickle model files totaling 1.87 GiB were excluded from the stabilized source release. Their paths, sizes, and SHA-256 hashes are retained in `archive/manifests/EXTERNAL_MODEL_ARTIFACTS.csv`.

## Gate decision

Level 1 is validated for its declared solver-independent scope. Level 2 analytical validation may begin without rewriting the preserved core. A licensed Gurobi baseline rerun remains a mandatory Level 2 entry check for optimization changes.
