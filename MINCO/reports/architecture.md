# MINCO Architecture Summary

MINCO currently implements a stabilized research-grade core over a synthetic three-hospital reference network.

```text
Validated synthetic inputs
  -> expected Markov patient-state forecast
  -> resource-pressure calculation
  -> Gurobi network capacity LP
  -> seeded stochastic state-transition evaluation
  -> risk classification, explanation, and recommendation
  -> FastAPI / Streamlit presentation boundaries
  -> SQLite audit records
```

The current simulator is not a discrete-event hospital twin. The uncertainty-adjusted policies are not formal robust optimization. These boundaries are deliberate and documented until Level 2 and Level 3 evidence gates are completed.

See `docs/architecture/CURRENT_ARCHITECTURE.md` and `docs/execution/MINCO_IMPLEMENTATION_ROADMAP_V1.0.md`.
