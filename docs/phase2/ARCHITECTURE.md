# Phase 2 — Native Hospital Operations Workstation Architecture

## Product identity

Phase 2 changes MINCO from a collection of analytical services into a hospital-oriented native Windows workstation. The primary interaction model is a census/flow/planning surface, not a generic web dashboard.

```text
Flutter / Dart Windows workstation
            │
            │ gRPC + Protocol Buffers
            ▼
Python operational runtime
 ├─ canonical hospital events
 ├─ DuckDB event lake / Parquet replay
 ├─ Polars historical analytics
 ├─ state reconstruction
 ├─ patient-level DES
 ├─ Markov / Monte Carlo / ML engines
 └─ adaptive Claude evidence review
            │
            │ typed stochastic instance
            ▼
Julia / JuMP / Gurobi
 ├─ extensive-form stochastic MILP
 ├─ CVaR tail-risk
 ├─ Progressive Hedging
 └─ rolling-horizon reoptimization
```

The older FastAPI surface remains compatibility infrastructure only. Flutter + gRPC is the V1 product path.

## Workstation surfaces

1. **Regional Census Board** — reconstructed operational state with capacity, occupancy, waiting and provenance.
2. **Patient Flow Theatre** — network-oriented operational flow view rather than chart cards.
3. **Monte Carlo Room** — matched-future comparison of baseline and intervention policies.
4. **Intervention Composer** — 72-hour action timeline, CVaR risk posture and direct Julia/JuMP/Gurobi plan execution.
5. **Decision Review Board** — deterministic evidence assembly plus cost-aware Claude Haiku/Sonnet routing.

## Data modes

- `SYNTHETIC`: bundled Meridian reference data.
- `FILE_BATCH`: canonical CSV ingestion.
- `HISTORICAL_REPLAY`: event lake replay through a selected as-of timestamp.
- `EXTERNAL_STREAM`: contract reserved for a future source adapter; no real hospital source is bundled.

Missing required canonical fields are rejected. MINCO does not synthesize absent hospital identifiers, event times, resource types or source provenance.

## Distinctive stack rationale

| Component | Technology | Why here |
| --- | --- | --- |
| Native client | Flutter / Dart | Windows-native operational workstation; avoids another browser dashboard |
| Service boundary | gRPC + Protobuf | Strong cross-language contract for Dart ↔ Python |
| Historical/replay lake | DuckDB + Parquet | Embedded analytical replay without requiring a database server |
| Transformation | Polars | Lazy columnar scans over Parquet event history |
| Computational AI | Python / scikit-learn + custom stochastic models | Existing validated ML and MMPP code |
| Primary OR | Julia / JuMP / Gurobi | Stochastic MILP, CVaR and decomposition are first-class mathematical components |
| Simulation | Python custom DES + Monte Carlo | Transparent patient conservation and common-random-number experimentation |
| LLM | Anthropic Claude | Evidence interrogation only, adaptively routed for cost |

## Evidence boundary

The native platform is validated against synthetic/replay events. It is not clinically validated, connected to an EHR, or approved for real hospital deployment.
