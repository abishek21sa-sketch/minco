> **RC3 Windows acceptance:** from the outer extracted RC3 folder, run `.\RUN_ACCEPTANCE.cmd`. Do not run the PowerShell acceptance scripts directly. The RC3 launcher creates/verifies a Python 3.14 project venv, installs the acceptance dependencies, and runs the correct repository gate without depending on PowerShell execution policy.

# MINCO — Stochastic Hospital Operations Decision Workstation

## Deployment

Deploy the repository root to Vercel using `vercel.json`; the build generates
the MINCO decision workstation at `/app`. Deploy the same repository as a
Render Blueprint, where the runtime binds to `$PORT` and exposes `/health`.
The Vercel artifact routes its `/api/*` calls to the Render service through a
configurable browser bridge; all outputs remain synthetic, human-reviewed
decision support.

MINCO is a computational healthcare-operations system for deciding how a hospital network should respond to uncertain patient flow **before capacity failure occurs**.

It is intentionally not a generic analytics dashboard. The mathematical identity of MINCO is:

```text
Poisson Hidden Markov / MMPP demand regimes
+ continuous-time Markov patient flow
+ discharge-hazard and ICU-escalation ML
+ queueing / conservation / staffed-capacity IE
+ common-random-number Monte Carlo
+ two-stage stochastic MILP + CVaR
+ Progressive Hedging
+ rolling-horizon reoptimization
```

The bundled Meridian reference network is synthetic. All Phase-1 results are **synthetic benchmark evidence**, not real-hospital or clinical claims.

## Phase 1 decision workflow

```text
UNCERTAIN ARRIVALS
      ↓
HIDDEN DEMAND REGIME (Poisson HMM / MMPP)
      ↓
DISCHARGE HAZARD + ICU ESCALATION ML
      ↓
CTMC PATIENT-FLOW / QUEUEING CONSEQUENCE
      ↓
STOCHASTIC ICU-CENSUS SCENARIOS
      ↓
JULIA / JuMP / GUROBI TWO-STAGE MILP + CVaR
      ↓
PROGRESSIVE HEDGING / ROLLING HORIZON
      ↓
MONTE CARLO POLICY STRESS TEST
      ↓
HUMAN-REVIEWED OPERATIONAL ACTION
```

## Real AI

Phase 1 contains three distinct probabilistic/ML capabilities:

1. a custom Poisson Hidden Markov model trained with Baum-Welch EM to infer latent demand regimes;
2. a discrete-time discharge hazard model for future bed releases;
3. a calibrated Gradient Boosting model for operational ICU escalation probability.

The latter two probabilities directly parameterize stochastic ICU-census scenarios used by optimization. Claude is **not** counted as the project's computational AI.

## Real Industrial Engineering

Executable mathematics includes continuous-time Markov patient flow, expected residence time, Erlang-C M/M/s queueing, Kingman variability approximation, Little's Law, patient-flow conservation and staffed-bed capacity.

## Real Operations Research

The primary V1 optimizer is **Julia + JuMP + Gurobi**. It is a two-stage stochastic hospital-network MILP with binary surge activation, integer flex staffing, integer transfer/deferral recourse, expected recourse cost and CVaR tail-risk. Progressive Hedging decomposes scenario problems; rolling horizon repeatedly replans as state information changes.

A separate Python/SciPy/HiGHS implementation plus exact enumeration validates small instances independently.

## Monte Carlo laboratory

Policies are compared on the **same stochastic futures** using common random numbers. Randomness includes demand regime transitions, arrivals, patient pathways, staff absence and capacity shocks. Results are explicitly labeled **SIMULATED**.

## Claude routing

MINCO uses Anthropic Claude only above deterministic evidence. A local router chooses a fast Haiku-family request for routine status/explanation work and a Sonnet-family request for cross-module decision review. It can downgrade to Haiku when the configured Sonnet daily token budget is exhausted. Model IDs can be pinned through environment variables or resolved through Anthropic's Models API. No API key is required for core engineering workflows.

## Phase-1 validation

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[analytics,dev]"
pytest
python -m src.system.run_phase1_mathematical_engine_validation
```

For the primary stochastic OR runtime on Windows:

```powershell
julia julia/setup.jl
```

A valid local Gurobi license is required for Julia/Gurobi execution. Credentials must never be committed.

## Repository map

- `src/stochastic` — Poisson HMM/MMPP, CTMC, Monte Carlo and AI-to-OR scenario generation
- `src/ml` — discharge-hazard and ICU-escalation ML
- `src/queueing` — Erlang-C, M/M/s, Kingman and Little's Law
- `src/decision_math` — conservation, staffed capacity, independent stochastic-MILP oracle, PH and rolling-horizon checks
- `julia/src` — primary JuMP/Gurobi extensive form, Progressive Hedging and rolling horizon
- `src/claude` — deterministic cost-aware Claude model routing
- `results/validation/phase1_mathematical_engine` — reproducible Phase-1 evidence
- `docs/phase1` — architecture and mathematical methods

The existing FastAPI/Streamlit work is retained only as historical compatibility infrastructure. It is **not** the locked V1 product architecture. Phase 2 replaces the product interface with a native Flutter/Dart Windows hospital workstation using gRPC/Protocol Buffers.

## Evidence boundary

Phase 1 is **VALIDATED ON SYNTHETIC GENERATIVE / BENCHMARK DATA**. Real-hospital calibration, clinical validation, production security approval and real deployment remain external validation work.

See `docs/phase1/PHASE1_MATHEMATICAL_METHODS.md` for equations, variables, assumptions, validation and implementation locations.

---

## Phase 2 — Native Hospital Operations Workstation

Phase 2 changes the primary product architecture to:

```text
Flutter/Dart Windows workstation
        ↓ gRPC + Protocol Buffers
Python operational runtime
        ↓
DuckDB + Parquet + Polars event/replay layer
        ↓
Markov + ML + DES + Monte Carlo
        ↓
Julia/JuMP/Gurobi stochastic optimization
```

The workstation is deliberately modeled after dense hospital operations software rather than a portfolio dashboard. Its five primary surfaces are **Census Board**, **Patient Flow Theatre**, **Monte Carlo Room**, **Intervention Composer**, and **Decision Review Board**.

The bundled configuration declares `MINCO_SOLVER_LICENSE_MODE=academic` for synthetic replay, research, and educational use. If the Julia primary solver is unavailable, the runtime automatically uses the independent SciPy/HiGHS verification oracle and labels the plan with its fallback provenance instead of failing the workstation request.

### Phase-2 core validation

```powershell
python -m src.system.run_phase2_core_validation
```

### Phase-2 Windows/data dependencies

```powershell
python -m pip install -e ".[phase2,analytics,solver,dev,llm]"
python -m src.system.run_phase2_backend_acceptance
```

### Native Flutter build

```powershell
.\scripts\bootstrap_flutter_windows.ps1
```

Then run the computational runtime and workstation in separate PowerShell terminals:

```powershell
.\scripts\start_phase2_runtime.ps1
.\scripts\start_workstation_windows.ps1
```

The Anthropic API key remains optional. Set `ANTHROPIC_API_KEY` in `.env` only when the external Decision Review Board call is desired. Deterministic engineering, simulation and optimization do not depend on Claude.


## Signature algorithm
See [`docs/SIGNATURE_ALGORITHM.md`](docs/SIGNATURE_ALGORITHM.md) for the governed FLOW-CVaR formulation and validation contract.

## Enterprise control tower

The current enterprise product-hardening milestone adds a unified operator read model over the existing validated services:

- `GET /v1/control-tower` exposes network state, predicted patient flow, FLOW-CVaR risk and actions, scenario catalog, solver diagnostics, evidence hashes, data freshness, and human-gated authorization in one versioned contract;
- `POST /v1/audit/{run_id}/review` records an immutable human disposition (`ACCEPTED_FOR_OPERATIONS_REVIEW`, `DEFERRED`, or `REJECTED`), with `GET /v1/audit/{run_id}/reviews` and `GET /v1/audit/reviews` for traceability;
- the Command Center **Operations Workbench** compares a baseline scenario with a counterfactual through `GET /v1/scenarios` and `POST /v1/scenarios/evaluate`, then routes the selected run into the same human-review and audit-integrity chain; comparison is decision support only and never dispatches an action;
- `GET /v1/decision-packets/{run_id}` composes the governed run, what-if metrics, immutable manifest, review history, audit-integrity result, and release evidence into a portable review packet; packet readiness never grants execution authority;
- `GET /v1/identity` exposes redacted caller identity, effective role, permissions, and authorization posture; deployment role policy supports `viewer`, `analyst`, `reviewer`, and `admin`, with reviewer/admin required to record a disposition;
- `GET /v1/audit/integrity` verifies the SHA-256 hash chain over human-review events and reports tampering without authorizing execution;
- `/health` and `/readiness` return typed system contracts, and every API response carries an `X-Correlation-ID` for cross-service incident tracing; structured request events omit query parameters and payloads;
- `GET /v1/observability/metrics` and `/metrics` expose low-cardinality request, error, route, and latency counters for SLO dashboards; payloads, query strings, and identifiers are not retained;
- when `MINCO_API_KEY` or `MINCO_API_KEYS` is configured, operational routes enforce constant-time API-key/Bearer authentication and minimum roles while liveness/readiness remain available for platform probes; credentials are never logged;
- the Phase 2 backend gate preflights the Julia project dependencies before selecting the primary solver, preserving an explicit SciPy/HiGHS fallback when Julia is installed but uninstantiated;
- `GET /v1/operational-history` exposes canonical replay/event-lake provenance, reconstructed network state, event counts, and freshness posture; replay data is explicitly not treated as a live ADT/EHR feed;
- `GET /v1/operational-events/status` and `POST /v1/operational-events/ingest` expose a bounded external-stream adapter with schema, provenance, freshness, deduplication, and de-identification guards; the adapter is available but remains disconnected until an approved ADT/EHR source is configured;
- `GET /v1/release-readiness` returns an evidence-backed deployment scorecard with typed checks, security posture, artifact references, explicit blockers, and next actions;
- `GET /v1/release-evidence` returns a hash-addressed release attestation over the contract registry, backend acceptance report, runtime evidence, manifest, and project status; this verifies package integrity without claiming production approval;
- `RUN_APP.cmd` serves the governed FastAPI API and executive Command Center on port 8814; `RUN_DEMO.cmd` remains available for the isolated deterministic legacy demo;
- `scripts/generate_release_evidence.py` writes the same attestation to `results/validation/release_evidence_attestation.json` for release-review handoff;
- `RUN_ENTERPRISE_ACCEPTANCE.cmd` runs the repeatable candidate-package acceptance harness; the operator procedure is documented in `docs/testing/ENTERPRISE_ACCEPTANCE_TESTING_PROCEDURE.md`;
- the Streamlit workstation includes an **Enterprise Control Tower** surface for command-center review;
- the control tower reports `HUMAN_REVIEW` or `BLOCKED` separately from the mathematical `AUTHORIZED` gate and never authorizes autonomous clinical, staffing, transfer, or diversion actions. Review events record disposition only; they do not dispatch actions.

See [`docs/architecture/ENTERPRISE_CONTROL_TOWER.md`](docs/architecture/ENTERPRISE_CONTROL_TOWER.md) for the operating model and release boundary.

## Scaled research twin

The current native runtime can run an enterprise-sized synthetic research
profile instead of the three-facility reference case:

- the default DuckDB replay profile targets 120 facilities and at least
  100,000 canonical lifecycle events;
- `scripts/refresh_public_hospital_reference.py` downloads the public CMS
  Hospital General Information directory into
  `data/public_reference/cms_hospital_general_information.csv`;
- CMS metadata is used only for facility identity/strata calibration. Bed
  capacity, occupancy, patient-flow events and outcomes remain synthetic;
- operating modes are `normal`, `lower_demand`, `respiratory_surge`,
  `covid_like`, `flu_like`, and `mixed_surge`; these are stress-test
  parameterizations, not clinical disease classifiers;
- `scripts/run_scaled_research_acceptance.py` writes a repeatable evidence
  report to `results/validation/scaled_research_twin/` and probes a 24-hospital
  stochastic optimization input.

```powershell
python scripts/refresh_public_hospital_reference.py --limit 10000
python scripts/run_scaled_research_acceptance.py
cmd /c RUN_NATIVE_WORKSTATION.cmd
```

This profile is suitable for academic research, replay and engineering
validation. It is not an operational hospital deployment: real ADT/EHR data,
local clinical validation, calibration, security review, reliability testing,
and human governance approval are still required before any live use.

## Next-level readiness documentation

The repository-wide roadmap is [`docs/NEXT_LEVEL_REQUIREMENTS.md`](docs/NEXT_LEVEL_REQUIREMENTS.md). It covers clinical scope, real-data integration, mathematics, ML/AI, privacy, security, platform engineering, reliability, human factors, deployment, licensing, and adoption. Each area includes required work and exit evidence.

The technical foundation is documented in [`docs/MINCO_MATHEMATICAL_AND_AI_FOUNDATIONS.md`](docs/MINCO_MATHEMATICAL_AND_AI_FOUNDATIONS.md). It explains the demand regimes, CTMC patient flow, Little’s Law, queueing, correlated scenarios, stochastic MILP, CVaR, simulation, ML targets and validation, generative-AI boundaries, and the AI → IE → OR coupling contract.
