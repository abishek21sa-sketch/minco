# MINCO Next-Level Requirements

## Purpose

This document is the product-wide readiness plan for MINCO. It converts the current research workstation into a system that can be evaluated by a large health-system technology, clinical-operations, security, and enterprise architecture review board.

The document is intentionally evidence-led. A feature is not considered ready because it exists in the UI; it is ready when its behavior, limitations, ownership, security posture, and failure modes are demonstrated with repeatable evidence.

## Current baseline

The current repository provides a strong research foundation:

- a 120-facility synthetic/replay network with 121,239 events;
- public CMS facility metadata used for facility strata and identifiers;
- aggregate CDC respiratory-surveillance references used only for directional mode calibration;
- six explicit operating modes: normal, lower demand, respiratory surge, COVID-like stress, flu-like seasonal peak, and mixed respiratory surge;
- a continuous-time Markov patient-flow model, queueing and conservation calculations, stochastic simulation, and a two-stage capacity optimizer with CVaR risk treatment;
- temporal ML validation for ICU-escalation risk with a held-out ROC-AUC of approximately 0.653 and a Brier score better than a constant baseline on the bundled research replay;
- a native Windows workstation, governed API, human-review workflow, audit hash chain, release evidence, API-key boundary, and runtime readiness checks;
- explicit synthetic-data, replay-only, human-review, and non-autonomous-use boundaries.

These are engineering and research capabilities. They are not evidence of clinical effectiveness, operational safety, or suitability for an individual hospital.

## Readiness model

MINCO should advance through five evidence gates:

1. **Research correctness** — equations, code paths, stochastic behavior, and ML evaluation are reproducible and independently checked.
2. **Data validity** — source data has documented ownership, semantics, freshness, quality thresholds, provenance, and permitted use.
3. **Operational validation** — historical replay and shadow-mode comparisons show calibrated performance across facilities, seasons, and stress conditions.
4. **Enterprise control** — identity, least privilege, privacy, auditability, availability, incident response, and change control are tested.
5. **Controlled adoption** — a named health-system owner approves a narrowly scoped workflow, success criteria, rollback plan, training plan, and monitoring period.

No later gate compensates for a failed earlier gate.

## Cross-functional requirements

### 1. Clinical and operations scope

**Required next steps**

- Define the exact operational decisions in scope: census visibility, capacity planning, transfer coordination, staffing scenarios, elective-flow planning, or another bounded workflow.
- Define decisions explicitly out of scope, including diagnosis, treatment selection, individual triage, automatic diversion, automatic staffing changes, and automatic patient movement.
- Establish a clinical-operations advisory group with accountable owners from nursing, emergency medicine, bed management, transfer center operations, infection prevention, quality, and informatics.
- Create a decision policy for every recommendation type: who may review it, what evidence is required, what happens when evidence is missing, and who owns the final action.
- Replace generic facility cards with configurable local resource taxonomies, service lines, bed definitions, staffing rules, transfer lanes, and operating calendars.

**Exit evidence**

- signed scope and non-scope document;
- workflow swimlanes showing MINCO input, recommendation, human decision, and downstream action;
- named decision owners and escalation paths;
- usability study with representative operators;
- documented rollback and downtime procedure.

### 2. Real data and interoperability

**Required next steps**

- Add an approved ingestion architecture for ADT, bed management, staffing, transfer, elective-surgery, ED, ICU, and supply-chain feeds where relevant.
- Define a canonical event contract with source timestamp, event timestamp, facility, encounter surrogate, service, resource, provenance, correction semantics, and late-arrival handling.
- Support HL7 v2, FHIR, secure file exchange, and database/API adapters through isolated connectors rather than embedding source-specific logic in the decision engine.
- Implement a data-quality service for completeness, timeliness, duplicates, impossible transitions, clock skew, unit mismatches, and reconciliation against source totals.
- Separate PHI-bearing raw zones from de-identified feature zones. Store only the minimum fields required by an approved use case.
- Version facility mappings, service definitions, code systems, and historical corrections.

**Exit evidence**

- data-use approval and source-owner signoff;
- source-to-canonical mapping specification;
- automated quality report with thresholds and alerting;
- replayable golden dataset with expected outputs;
- provenance for every feature used by a forecast or optimization run.

### 3. Mathematical decision engine

**Required next steps**

- Formalize the full model in a versioned mathematical specification with units, domains, constraints, objective terms, and assumption IDs.
- Add independent small-instance oracles for each optimization family and compare primal feasibility, objective value, dual information where available, and action monotonicity.
- Add explicit constraint tests for transfer conservation, bed conservation, staffing feasibility, service-line eligibility, time windows, and network flow balance.
- Upgrade the scenario generator from hand-calibrated multipliers to a documented hierarchical model that estimates facility-level and network-level dependence from approved historical data.
- Add out-of-sample stress tests for missing facilities, closed units, transfer-lane failure, delayed data, extreme arrival bursts, and solver time limits.
- Specify solver tolerances, reproducibility seeds, timeout behavior, incumbent-plan policy, and fallback semantics.
- Add model-risk review for objective weights so that cost, unsafe excess, overflow, transfer burden, staffing burden, and equity impacts cannot be tuned without traceability.

**Exit evidence**

- signed mathematical specification;
- independent solver/oracle comparison report;
- constraint coverage report;
- scenario-dependence validation;
- deterministic rerun evidence and solver failure evidence;
- documented interpretation of every output shown to operators.

### 4. ML and AI system

**Required next steps**

- Define target labels, prediction horizons, observation windows, censoring rules, exclusion rules, and permissible feature availability at prediction time.
- Use facility-aware, time-aware, and prospective holdouts. Do not rely on random row splits for operational claims.
- Establish performance thresholds by facility size, service line, operating mode, season, and data-quality tier.
- Report calibration, discrimination, precision-recall behavior, decision-curve utility, alert burden, and abstention behavior rather than a single headline score.
- Add model cards, feature cards, training manifests, data snapshots, dependency locks, and signed artifact hashes.
- Add drift detection for volume, acuity, missingness, coding patterns, calibration, and residuals.
- Create an abstain-and-escalate path when features are stale, the model is outside its training envelope, confidence is low, or the data contract fails.
- Keep generative AI in a bounded explanation and retrieval role. It must not alter mathematical constraints, silently rewrite recommendations, or exercise operational authority.
- Add prompt/version tracing, retrieval-source citations, output filtering, adversarial tests, and a no-PHI policy for any optional language-model feature.

**Exit evidence**

- model cards and validation reports;
- prospective or shadow-mode evaluation;
- subgroup and facility-stratified performance;
- calibration and drift dashboards;
- documented abstention rates and operator override rates;
- red-team report for generative features;
- signed model and prompt release manifest.

### 5. Data governance, privacy, and compliance

**Required next steps**

- Determine whether the deployment is subject to HIPAA, state privacy law, health-system policy, research IRB requirements, or contractual data-use restrictions.
- Establish data classification, retention, deletion, legal hold, access review, and breach-response procedures.
- Use tokenization or surrogate identifiers and keep re-identification keys outside the analytics environment.
- Encrypt data in transit and at rest; use managed secrets and key rotation rather than environment variables in production.
- Implement immutable audit events for data access, model execution, configuration changes, reviewer actions, and export/download operations.
- Complete threat modeling for PHI exposure, tenant isolation, insider misuse, prompt injection, model poisoning, replay tampering, and supply-chain compromise.

**Exit evidence**

- privacy impact assessment;
- security architecture and threat model;
- access-control matrix and quarterly access review;
- penetration test and remediation record;
- retention/deletion test;
- incident-response tabletop exercise.

### 6. Platform and software engineering

**Required next steps**

- Package API, runtime, worker, solver, UI, and connector services as versioned deployable units.
- Replace the local-process launcher with a supported service supervisor, health checks, graceful shutdown, dependency readiness, and restart policies.
- Add database migration tooling, backup/restore, point-in-time recovery, and tested disaster-recovery procedures.
- Introduce CI/CD gates for formatting, static analysis, unit tests, contract tests, integration tests, security scanning, dependency licenses, container scanning, and reproducible builds.
- Separate configuration from code and validate every configuration change against a schema.
- Add feature flags for new algorithms and a kill switch that disables recommendation publication without deleting evidence.
- Define API compatibility policy, deprecation windows, and client version negotiation.

**Exit evidence**

- deployable environment manifests;
- tested backup and restore;
- build provenance and software bill of materials;
- API contract compatibility report;
- failure-injection results;
- rollback rehearsal.

### 7. Reliability and observability

**Required next steps**

- Define service-level objectives for freshness, availability, p95/p99 latency, optimization completion, evidence persistence, and recovery time.
- Emit traces across ingestion, feature generation, forecast, scenario generation, optimization, simulation, review, and export.
- Monitor data freshness separately from service health. A healthy API with stale clinical data must be treated as degraded.
- Add alerts for missing sources, event lag, schema drift, facility drop-off, model drift, solver fallback, elevated abstention, and audit-chain errors.
- Maintain an operator-visible status page and an incident timeline with correlation IDs.
- Test graceful degradation: read-only replay, last-known-good plan, safe abstention, and complete recommendation blackout.

**Exit evidence**

- SLO definitions and error budgets;
- dashboards and alert runbooks;
- synthetic monitoring;
- load, soak, and failover tests;
- recovery-time and recovery-point evidence;
- incident-response exercise.

### 8. User experience and human factors

**Required next steps**

- Show source freshness, model scope, confidence/calibration, key assumptions, scenario definition, solver status, constraints, and unresolved warnings beside every recommendation.
- Make “why this recommendation,” “what changed,” “what could invalidate it,” and “what happens if I reject it” first-class interactions.
- Support keyboard navigation, screen readers, contrast, zoom, responsive layouts, and accessible error states.
- Add role-specific workspaces for executive overview, capacity analyst, transfer center, staffing, clinical reviewer, and system administrator.
- Provide exportable review packets with data snapshot, model versions, scenario set, objective terms, recommendation, reviewer identity, and decision rationale.
- Test cognitive load, alert fatigue, false reassurance, and interpretation of uncertainty with representative users.

**Exit evidence**

- accessibility conformance report;
- task-based usability results;
- human-factors risk register;
- operator training and competency checklist;
- review-packet sample approved by stakeholders.

### 9. Deployment and adoption

**Required next steps**

- Start with a silent or shadow deployment that produces predictions without influencing operations.
- Compare forecasts, capacity estimates, recommendations, and outcomes against existing workflows and strong baselines.
- Use a staged rollout: one service, one facility, a small network cohort, then expansion only after predefined gates pass.
- Define success metrics that include operational outcomes, workload, fairness, override rate, safety events, and user trust—not only model scores.
- Create training, support, ownership, release communication, change-management, and end-of-life plans.
- Document licensing for all solvers, libraries, data, models, and optional AI providers. Academic solver licensing must remain isolated from commercial operational use unless the license explicitly permits it.

**Exit evidence**

- shadow-mode report;
- baseline comparison;
- staged rollout approval;
- go/no-go checklist;
- training completion;
- support and ownership agreement;
- license review.

## Prioritized execution plan

### P0 — Required before external operational evaluation

- finalize use-case and non-use-case boundaries;
- obtain approved data owners and data-use permissions;
- build the canonical event and data-quality contracts;
- complete threat model, privacy review, and access-control design;
- formalize the mathematical specification and independent solver checks;
- define temporal/facility-aware ML validation and abstention rules;
- establish SLOs, runbooks, rollback, and shadow-mode evaluation;
- remove any dependency on academic-only licensing for a commercial deployment path.

### P1 — Required before broader pilot

- integrate at least one approved real operational source;
- demonstrate historical replay and prospective shadow performance;
- validate calibration and drift across facilities and operating modes;
- complete load/failover/disaster-recovery testing;
- complete accessibility and human-factors testing;
- produce signed model, data, software, and release manifests;
- deliver role-based review packets and operator training.

### P2 — Required for durable enterprise operation

- support multi-tenant or multi-network isolation where applicable;
- add automated champion/challenger evaluation and safe model promotion;
- provide capacity, staffing, transfer, and elective-flow extensions with local policy controls;
- establish independent periodic model-risk review;
- benchmark cost, latency, carbon/compute footprint, and operational benefit;
- maintain a public-safe research release and a separately governed operational distribution.

## Definition of done for the next release

The next release is complete only when all of the following are true:

- the README links to this roadmap and the mathematical/AI foundations;
- every production-bound claim has an evidence owner and an acceptance test;
- synthetic and real-data paths are visibly distinct;
- stale or invalid data causes abstention or an explicit degraded state;
- optimizer, simulator, and ML outputs are reproducible from a versioned manifest;
- human review remains mandatory for in-scope operational decisions;
- access, audit, privacy, retention, backup, and incident controls are tested;
- a named deployment owner approves the scope, success metrics, rollback plan, and pilot boundary.

## Current claim boundary

MINCO currently demonstrates a scalable, governed research workstation over synthetic/replay data with public aggregate calibration references. It does not yet demonstrate clinical effectiveness, real-time hospital reliability, patient-level prediction validity, or authorization for autonomous action. The requirements above are the work needed to establish those claims responsibly.
