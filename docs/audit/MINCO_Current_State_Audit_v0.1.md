# MINCO Current-State Audit v0.1

**Audit scope:** `configs.zip` received July 27, 2026  
**Archive SHA-256:** `a79301737ddfa77253a8d2b5f68b310c4777810dee168a875c0f3eb92e4a1a51`  
**Archive entries:** 894  
**Archive period represented by ZIP timestamps:** April 9, 2026 through June 22, 2026  
**Governing target:** MINCO Engineering Specification v1.0  
**Roadmap status:** Not produced. No completion percentage or future phase assignment is included.

## 1. Audit conclusion

The supplied archive contains a substantial and technically meaningful **healthcare operations research and decision-intelligence prototype**. It is not an empty project and should not be restarted.

The strongest existing core is:

- a synthetic three-hospital network model,
- four patient cohorts with Markov transition matrices,
- deterministic demand forecasting,
- seeded stochastic state-transition simulation,
- continuous linear optimization for surge, elective-control, and transfer decisions,
- nominal, uncertainty-adjusted, and regime-adjusted policy variants,
- scenario and sensitivity experiments,
- supervised prediction experiments,
- policy explanations and counterfactual summaries,
- a Streamlit command-center interface,
- a FastAPI design,
- an SQLite audit trail,
- extensive stored result tables and figures.

The archive does **not** yet constitute the enterprise-grade MINCO platform described by Specification v1.0. The most important missing or defective capabilities are automated testing, clean reproducibility, operational data ingestion, real state reconstruction, discrete-event simulation, formal robust/stochastic optimization, production API packaging, security, observability, deployment, CI/CD, and real-world validation.

## 2. Audit scope and limitations

This audit is complete for the evidence contained in the supplied archive. It does not claim that omitted local files, Git history, private data, credentials, or external experiment records do not exist.

The audit environment did not contain `gurobipy` or `streamlit`. Therefore:

- the data loader, deterministic forecast, and seeded stochastic simulator were executed directly;
- Python syntax compilation was verified for all 134 Python files;
- pytest was executed and collected no tests;
- the Gurobi optimizer, live dashboard, command center, and full pipeline were not independently rerun;
- stored logs, source code, result tables, and metrics were cross-checked instead;
- model pickle files were not loaded during the audit.

## 3. Current-State Inventory

- 894 archive entries.
- Approximately 2.05 GB uncompressed.
- 134 Python source files, including 35 zero-byte Python files.
- 92 compiled `.pyc` files included in the archive.
- 362 CSV files.
- 58 pickle model files totaling approximately 2.01 GB.
- 116 PNG result figures.
- 22 JSON files.
- 8 Markdown files.
- 1 SQLite audit database.
- 4 empty notebooks.
- 4 empty test files.
- Root-level README, requirements file, makefile, two one-off edit scripts, an empty `main.py`, and a manual `test.py`.

The detailed file inventory is included in `MINCO_Archive_Inventory_v0.1.csv`.

## 4. Repository and Architecture Assessment

### Preserved architecture

The repository already separates substantial analytical responsibilities into:

- `src/config`
- `src/markov`
- `src/simulation`
- `src/optimizer`
- `src/optimization`
- `src/ai`
- `src/analysis`
- `src/agent`
- `src/command_center`
- `src/storage`
- `src/api`
- `src/rl`
- `src/scenarios`
- `dashboard`
- `experiments`
- `results`

This structure is worth preserving.

### Architectural defects

- The dashboard directly imports solver and simulation code rather than consuming stable service contracts.
- The primary dashboard is a 1,485-line Streamlit module.
- `src/api/main.py` is empty even though the documented command is `uvicorn src.api.main:app`.
- The FastAPI application is implemented in `src/api/__init__.py`.
- Several planned packages contain empty files.
- The repository mixes source, generated models, figures, logs, publication outputs, and compiled bytecode.
- MINCO and Meridian branding are inconsistent.
- The repository does not implement the specification's ingestion, state-repository, feature-store, security, observability, or deployment layers.

**Assessment:** Partially Implemented — High confidence.

## 5. Execution and Reproducibility Assessment

### Directly executed during audit

- All nonempty Python files parsed successfully.
- `compileall` completed successfully.
- The base healthcare instance loaded successfully.
- The deterministic Markov forecast executed and reproduced identical outputs.
- The seeded stochastic simulator executed and reproduced identical arrivals and state trajectories.
- pytest executed but collected no tests.

### Historical execution evidence

A stored system pipeline log dated May 6, 2026 records all nine pipeline stages as successful. The recorded wall-clock duration is approximately 10.5 hours, with the current-state calibrated regime model stage consuming approximately 10.4 hours.

### Reproducibility limitations

- Dependencies are unpinned.
- No Python version is declared.
- No environment lock is supplied.
- No Gurobi version or license setup is documented.
- Model registries do not include package versions, data hashes, model hashes, or signatures.
- Stored model paths use Windows separators.
- No clean-environment execution log is included.

**Assessment:** Previously Executed/Validated — Medium confidence. Clean reproducibility remains Unverified — High confidence.

## 6. Data Assessment

### What exists

The base instance contains:

- 3 hospitals,
- 4 patient cohorts,
- 7 modeled days,
- 84 arrival rows,
- 6 capacity rows,
- 6 safe-threshold rows,
- 6 surge-capacity rows,
- 6 directed transfer lanes,
- 21 elective-bound rows,
- 4 transition matrices with 25 state-transition rows each.

Synthetic scale instances exist for 10, 25, 50, 100, and 200 hospitals.

### What is validated

- Required columns exist in the base tables.
- Required fields have no missing values.
- Transition probabilities are bounded and each supplied transition row group sums to approximately 1.
- The supplied n10 transfer-lane table contains two duplicated directed lane rows.

### What is missing

- No operational source integration.
- No data dictionary.
- No field-level provenance.
- No lineage.
- No dataset version manifest.
- No duplicate validation in the loader.
- No identifier referential-integrity validation.
- No timestamp ordering or freshness checks.
- No schema-drift mechanism.
- No feature-store service.
- No real hospital data.

The `literature_calibrated.json` file explicitly describes itself as a template and instructs the user to replace values with exact estimates and references.

**Assessment:** Synthetic data foundation Implemented — High confidence. Enterprise data platform Partially Implemented — High confidence. Literature calibration Unverified — High confidence.

## 7. Prediction Assessment

### Implemented

- Markov patient-state forecasting.
- Unsafe-excess regression.
- Blocked-arrival regression.
- Utilization-critical classification.
- Unsafe-risk threshold classification.
- Regime assignment and lagged regime prediction.
- Model registry CSVs and serialized pipelines.
- Feature-importance and classification reports.

### Evidence

The strongest stored leakage-filtered regression result is next-period blocked-arrival forecasting with Extra Trees:

- R²: 0.839
- MAE: 1.915

Exact unsafe-excess magnitude prediction is weak:

- best R²: approximately 0.167
- MAE: approximately 5.363

Lagged calibrated-regime forecasting is limited:

- best macro-F1: approximately 0.423

### Validation defect

The forecasting scripts use random train/test splitting. For next-period operational prediction, this does not establish temporal or scenario-level generalization and may allow correlated rows from the same experiment families to appear in both train and test sets.

**Assessment:** Implemented — High confidence. Operationally Validated — No; Partially Implemented validation — High confidence.

## 8. Simulation Assessment

The current simulation is a seeded Monte Carlo cohort state-transition model. It loops by day, hospital, and cohort and samples state transitions from Markov matrices.

It is:

- stochastic,
- reproducible under fixed seeds,
- capable of generating state trajectories and capacity pressure,
- capable of repeated policy evaluation,
- useful as a benchmark simulator.

It is not:

- a discrete-event simulation,
- a patient-level queueing model,
- a real-time event processor,
- a continuously synchronized operational twin,
- a state-reconstruction engine built from ADT events,
- historically calibrated against real waiting-time, occupancy, or throughput distributions.

**Assessment:** Stochastic simulator Validated — High confidence. Specification-level digital twin Partially Implemented — High confidence.

## 9. Optimization Assessment

### Implemented

The core optimization model controls:

- surge capacity,
- unsafe-capacity slack,
- overflow slack,
- accepted/rejected elective admissions,
- ICU transfer flows.

All decision variables are continuous. The core model is a linear program, not a MILP.

### Verified result claims

The stored baseline scenario with 50 replications supports:

- unsafe excess: 7.59 under `no_control` versus 4.41 under `optimized_network`, a 41.897% reduction;
- maximum utilization: 1.166 versus 1.037, an 11.063% reduction.

The regime-adjusted policy has lower mean unsafe excess than nominal optimization in 4 of 5 supplied scenarios. This is a correct descriptive claim. The supplied scenario-level significance table does not establish statistically significant superiority for those differences.

The scalability table reports for 200 hospitals:

- 14,000 variables,
- 12,600 constraints,
- Gurobi runtime: 0.049 s,
- wall-clock solve time: 0.059 s,
- model-build time: 0.826 s,
- LP status recorded as optimal.

The safe claim is that **reported Gurobi solver runtime was 49 ms**. End-to-end model construction and solution were not below 50 ms.

### Robustness limitation

The robust and regime-robust wrappers increase or adjust arrival inputs and then solve the nominal LP. This is defensible as uncertainty-adjusted optimization, but it is not a formal robust optimization model with an uncertainty set or adversarial inner problem.

**Assessment:** Core LP Implemented — High confidence. Stored numerical evidence Validated — High confidence. Formal robust optimization Partially Implemented — High confidence.

## 10. Decision Intelligence Assessment

Implemented components include:

- policy comparison,
- weighted decision scoring,
- risk classification,
- recommendation generation,
- manager-facing explanations,
- scenario counterfactuals,
- alert generation,
- executive brief generation,
- an audit database.

The recommendation layer is largely deterministic and rule/template based. It has not been validated with hospital operators or subject-matter experts.

**Assessment:** Implemented prototype — High confidence. Operational recommendation validity Unverified — High confidence.

## 11. Software Engineering Assessment

### Strengths

- Domain-oriented module structure.
- Dataclass-based instance wrappers.
- Reusable live what-if function.
- Shared risk classification.
- Dedicated audit repository.
- Make targets for primary workflows.
- Clear experiment and result organization.

### Defects

- Broken API entry point.
- Empty root entry point.
- Empty tests.
- Empty notebooks and package modules.
- Unpinned dependencies.
- No package metadata or lockfile.
- No lint/type/static-analysis configuration.
- No CI.
- No Docker.
- No security layer.
- No observability.
- Large generated artifacts mixed with source.
- Compiled bytecode included.
- Direct frontend-to-solver coupling.
- Windows-specific paths in model registries.
- Inconsistent naming.

**Assessment:** Partially Implemented — High confidence.

## 12. Testing and Validation Assessment

### Existing validation evidence

- 50- and 200-replication Monte Carlo summaries.
- Sensitivity analysis.
- Named scenario comparisons.
- Approximate significance testing and bootstrap intervals.
- Scalability benchmark.
- Model-performance reports.
- Stored pipeline logs.
- Seed-controlled simulation.

### Missing validation

- Unit tests.
- Integration tests.
- End-to-end automated tests.
- API tests.
- Optimizer feasibility and constraint tests.
- Regression/golden-result tests.
- Data-contract tests.
- Historical simulation validation.
- Temporal ML validation.
- External validation.
- Subject-matter review.
- Deployment/performance tests.

**Assessment:** Experimental validation Partially Implemented — High confidence. Automated testing Broken — High confidence.

## 13. Documentation and Deployment Assessment

### Documentation present

- README.
- Architecture report.
- Executive summary.
- Resume bullets.
- Manuscript package and safe-claim guidance.
- Result figures and tables.

### Documentation defects

- README link to `reports/industry_demo/architecture.md` is broken.
- No data dictionary.
- No API guide.
- No testing guide.
- No deployment guide.
- No user guide.
- No engineering decision records.
- No changelog.
- No license or contribution guide.

### Deployment

No deployment implementation was found.

**Assessment:** Documentation Partially Implemented — High confidence. Deployment Planned — High confidence.

## 14. Specification Compliance Summary

The complete 50-section matrix is included in `MINCO_Specification_Compliance_Matrix_v0.1.csv`.

The repository is strongest against Sections 11, 18, 21, 22, 27, 41, and 43. It is weakest or broken against Sections 25, 29–30, 32, 34–37, 39–40, and 44.

## 15. Technical Debt Register

The detailed register is included in `MINCO_Technical_Debt_Register_v0.1.csv`.

Highest-priority debt:

1. Empty automated test suite.
2. Broken API entry point.
3. Unlocked environment and undocumented Gurobi requirements.
4. Synthetic-only data and unverified literature calibration.
5. Misalignment between digital-twin/robust-optimization terminology and implementation.
6. Random-split forecasting validation.
7. Repository bloat and ungoverned model artifacts.
8. No deployment, security, observability, or CI/CD.
9. Inconsistent MINCO/Meridian identity.
10. Duplicate and possibly mislabeled result artifacts.

## 16. Risk and Blocker Register

The detailed register is included in `MINCO_Risk_Blocker_Register_v0.1.csv`.

The immediate blockers to trustworthy continued execution are:

- no test safety net,
- no locked runnable environment,
- Gurobi installation/license dependency,
- broken API packaging,
- synthetic-only evidence,
- terminology that overstates current implementation fidelity.

## 17. Preservation Decisions

### Preserve

- Base instance schema and dataclasses.
- CSV loader and validation logic as a starting point.
- Markov forecast engine.
- Transition sampler and seeded stochastic simulator.
- LP model formulation, constraints, objectives, and solution parsing.
- Synthetic network generator and scalability benchmark.
- Scenario and sensitivity experiment logic.
- Statistical analysis code after test coverage is added.
- Live what-if core.
- Risk classification single source of truth.
- Counterfactual and policy explanation modules.
- Command-center alert and audit concepts.
- SQLite audit repository.
- Stored result tables that can be tied to reproducible run manifests.
- Safe-claim and limitation documents.

### Repair or refactor

- API package and entry point.
- Repository packaging.
- Dashboard/service separation.
- Configuration management.
- ML validation design.
- Model registry metadata.
- Full pipeline profiles and runtime.
- Digital-twin terminology and state model.
- Robust-optimization terminology/formulation.
- Documentation links and repository identity.
- Data validation and provenance.

### Remove or archive

- `.pyc` files.
- Empty notebooks and unused empty modules unless linked to explicit work items.
- One-off edit scripts after verification.
- Duplicate result artifacts.
- Large pickle files from the main source repository; retain them in a governed artifact store or release package.
- Superseded model variants without registry status.

### Do not preserve as validated claims

- “Literature calibrated” until references and parameters are supplied.
- “Discrete-event digital twin.”
- “Formal robust optimization.”
- “Production-ready API.”
- “Deployment-ready platform.”
- “General AI Copilot.”
- “Sub-50 ms end-to-end execution.”

## 18. Gap Analysis

The current repository is a strong analytical prototype centered on a synthetic hospital network. The target specification is an enterprise operational platform.

The principal gaps are:

- static CSV inputs versus operational ingestion,
- synthetic initial conditions versus live state reconstruction,
- cohort Markov simulation versus event-driven patient flow,
- uncertainty-adjusted nominal LP versus formal robust/stochastic optimization,
- random-split ML experiments versus temporal production forecasting,
- rule-based copilot versus grounded conversational analytics,
- local Streamlit execution versus service-oriented deployment,
- stored result evidence versus automated regression validation,
- local files versus versioned data/model artifacts,
- conceptual security/observability versus implemented controls.

## 19. Recommended Immediate Stabilization Actions

These are stabilization actions, not a future implementation roadmap.

1. Freeze the supplied archive and registers as the audit baseline using hashes.
2. Add a canonical repository identity and correct MINCO/Meridian naming.
3. Create a locked environment with Python, package, Gurobi, and license requirements.
4. Repair `src/api/main.py` and verify `/health` through an automated smoke test.
5. Implement minimum tests for loader validation, transition normalization, deterministic forecasting, seeded simulation, small-model feasibility, claim calculations, and API import.
6. Separate a fast smoke pipeline from standard and full-research pipelines.
7. Add a data dictionary, synthetic-generation specification, and provenance manifest.
8. Correct README claims and attach exact evidence files and statistical qualifications.
9. Rename the current twin and robust policies truthfully until stronger formulations are implemented.
10. Move generated models, figures, bytecode, and duplicate results out of the source tree into governed artifacts.
11. Rerun the baseline after stabilization and compare metrics against the archived evidence.
12. Submit the stabilized audit evidence for acceptance before any implementation roadmap is created.

## 20. Audit disposition

**Current repository disposition:** Preserve and stabilize. Do not restart.

**Roadmap authorization:** Withheld until the audit is accepted and the immediate evidence-preservation and critical stabilization defects are resolved or explicitly accepted as known debt.
