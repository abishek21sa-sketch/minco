# MINCO Current Architecture — Stabilized Core

MINCO is the product/repository identity. The bundled Meridian Health Network is the synthetic demonstration instance.

```text
Synthetic CSV inputs
      -> validated HealthcareInstance
      -> Markov expected-value forecast
      -> capacity-pressure assessment
      -> uncertainty-adjusted Gurobi LP policies
      -> seeded stochastic state-transition evaluation
      -> risk classification and explanation
      -> FastAPI service boundary / Streamlit demonstration UI
      -> SQLite decision audit records
```

## Current fidelity boundary

The simulator is a cohort-level stochastic state-transition model. It is retained because it is useful and reproducible, but it is not represented as a discrete-event or continuously synchronized operational twin.

The robust policy wrappers alter demand assumptions before solving the nominal LP. They are retained as uncertainty-adjusted policies and are not represented as formal robust optimization until the mathematical formulation is upgraded.
