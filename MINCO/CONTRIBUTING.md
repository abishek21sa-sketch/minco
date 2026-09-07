# Contributing to MINCO

MINCO uses evidence-gated engineering. A contribution is not complete because code runs once; it must include tests, documentation, assumptions, failure behavior, and evidence appropriate to its claims.

## Required contribution contents

1. A clearly scoped issue or work package.
2. Tests covering success and failure behavior.
3. Updated documentation and configuration contracts.
4. Reproducible commands and generated evidence manifests.
5. No patient-identifiable or institution-confidential data.
6. Accurate labels: Implemented, Validated, Partially Implemented, Planned, Proposed, Broken, Obsolete, or Unverified.
7. An architecture decision record for material interface, model, solver, or data-contract changes.

Run before opening a pull request:

```bash
python -m src.system.run_smoke_pipeline
pytest
```

Solver-backed changes must additionally pass the licensed Gurobi validation gate.
