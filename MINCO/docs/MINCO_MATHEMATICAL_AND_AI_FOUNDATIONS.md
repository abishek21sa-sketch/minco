# MINCO Mathematical, ML, and AI Foundations

## 1. Scope and design principle

MINCO is a decision-support system for hospital-capacity research. Its central design principle is that machine learning estimates uncertain quantities, industrial-engineering mathematics translates those quantities into flow and capacity relationships, operations research selects a constrained plan, and stochastic simulation stress-tests the plan before a human reviews it.

The system is therefore not a single black-box model. It is a chain of typed transformations:

```text
events and covariates
        ↓
calendar-time demand and risk estimates
        ↓
cohort-level patient-flow mathematics
        ↓
correlated stochastic scenarios
        ↓
capacity and recourse optimization
        ↓
discrete-event / Monte Carlo stress tests
        ↓
evidence packet and human review
```

The current implementation uses synthetic/replay data, public aggregate calibration references, and an academic solver posture. Every equation and metric below must be re-estimated and revalidated against approved local data before an operational claim is made.

## 2. Notation

| Symbol | Meaning |
|---|---|
| (h in H) | facility or hospital |
| (r in R) | resource class, such as ED, Ward, or ICU |
| (t in T) | discrete planning interval |
| (s in S) | stochastic demand scenario |
| (c in C) | patient cohort or acuity class |
| (A_{h,c,t}) | arrivals for facility (h), cohort (c), interval (t) |
| (B_{h,r,t}) | staffed capacity of resource (r) |
| (O_{h,r,t}) | occupancy |
| (W_{h,r,t}) | waiting or blocked demand |
| (x) | first-stage decisions made before the scenario is known |
| (y_s) | scenario-specific recourse decisions |
| (C_s(x,y_s)) | scenario cost or loss |
| (alpha) | CVaR confidence level |
| (eta) | CVaR auxiliary threshold variable |

All time quantities must carry an explicit unit. In MINCO, arrival counts are interval counts, residence times are hours or intervals, and capacities are staffed resource units. Unit tests should reject silent mixing of hours, days, and six-hour planning buckets.

## 3. Event and cohort representation

The replay and ingestion boundary converts source records into canonical operational events. An event should have:

- an immutable event identifier;
- source and event timestamps;
- facility and service/resource identifiers;
- an encounter or cohort surrogate rather than a raw patient identifier;
- event type and state transition;
- source provenance, schema version, and correction status.

The current scaled research twin creates 120 facility streams and more than 100,000 deterministic replay events. The public CMS source supplies facility metadata and strata; it does not supply observed bed capacity or patient-level outcomes. Patient-level flows, capacities, arrivals, and outcomes remain synthetic.

For modeling, events are aggregated into cohorts such as:

- ED arrival and service state;
- ward admission;
- ICU admission or escalation;
- discharge-ready or discharge;
- transfer request, acceptance, departure, and completion;
- elective or planned demand.

Aggregation is useful for capacity planning, but it destroys information. A production data contract must specify what is aggregated, what is retained for audit, and how late or corrected events alter the feature snapshot.

## 4. Demand and operating-regime mathematics

### 4.1 Poisson and overdispersed arrivals

The simplest arrival model for a facility/cohort/time cell is:

$$
A_{h,c,t} \sim \operatorname{Poisson}(\lambda_{h,c,t} \Delta t)
$$

where (lambda) is the arrival intensity and (Delta t) is the interval length. A Poisson model assumes conditional independence and equal mean and variance. Hospital arrivals frequently exhibit overdispersion, autocorrelation, day-of-week effects, and cross-facility dependence, so this equation is a baseline rather than a complete model.

### 4.2 Markov-modulated Poisson process

MINCO represents changing pressure through a latent regime (Z_t), for example low, normal, or surge:

$$
P(Z_{t+1}=j \mid Z_t=i)=P_{ij}
$$

and

$$
A_t \mid Z_t=k \sim \operatorname{Poisson}(\lambda_k \Delta t).
$$

The transition matrix (P) controls persistence. A high diagonal value means that a surge tends to last rather than disappear after one interval. In continuous time, the regime process is represented by a generator (Q_Z) whose off-diagonal entries are transition rates and whose rows sum to zero.

For a network, a common latent regime can create correlated pressure:

$$
\lambda_{h,c,t}=\exp(\beta_{h,c}^{\mathsf T}u_t + b_{h,c} + g(Z_t) + \epsilon_{h,t})
$$

where (u_t) contains calendar and observable operational covariates, (b_{h,c}) is a facility/cohort effect, and (epsilon_{h,t}) captures residual variation. A hierarchical fit is preferable to assigning the same multiplier to every hospital.

### 4.3 Operating modes

The current runtime exposes six named research modes. They are parameterizations of demand, ICU mix, length of stay, ED pressure, and transfer availability. They are not disease classifiers and do not infer a patient's diagnosis.

For mode (m), the scenario intensity can be written as:

$$
\lambda^{(m)}_{h,c,t}=d_m\,e_{m,c}\,\lambda^{(0)}_{h,c,t}
$$

and expected residence time as:

$$
\tau^{(m)}_{c}=\ell_m\,\tau^{(0)}_{c}.
$$

The multipliers (d_m), (e_{m,c}), and (ell_m) are currently synthetic stress-test parameters directionally informed by public aggregate respiratory trends. A local deployment would estimate them from approved historical data and publish uncertainty intervals.

## 5. Patient-flow continuous-time Markov chain

### 5.1 Transient states and generator

For one cohort, let transient states be (E) (ED), (W) (Ward), and (I) (ICU). Let (D) be an absorbing discharge state. A continuous-time Markov chain has generator:

$$
Q=\begin{bmatrix}
-(\mu_E+\rho_{EW}+\rho_{EI}) & \rho_{EW} & \rho_{EI} \\
0 & -(\mu_W+\rho_{WI}) & \rho_{WI} \\
0 & 0 & -\mu_I
\end{bmatrix}
$$

where:

- (mu_E,mu_W,mu_I) are discharge or completion rates;
- (ho_{EW}) is ED-to-Ward transition rate;
- (ho_{EI}) is ED-to-ICU escalation rate;
- (ho_{WI}) is Ward-to-ICU escalation rate.

The diagonal is the negative sum of all outgoing rates. If return flows, transfers, or competing absorbing outcomes are needed, the matrix gains additional states and transitions.

### 5.2 Expected time in each resource

Let (Q_T) be the transient-state submatrix. The fundamental matrix is:

$$
N=(-Q_T)^{-1}.
$$

For initial transient distribution (pi_0), expected time in each transient state is:

$$
\mathbb{E}[\mathbf{T}]=\pi_0 N.
$$

If the interval unit is hours, (mathbb{E}[T_r]) is expected resource-hours per arriving patient. For a cohort with expected arrivals (a_c), expected resource demand is:

$$
D_{c,r}=a_c\,\mathbb{E}[T_{c,r}].
$$

This is the bridge between probabilistic flow and capacity planning. It is more informative than predicting length of stay as one scalar because it preserves resource-specific residence time.

### 5.3 Little's Law closure

For a stable flow system:

$$
L=\lambda W
$$

where (L) is average number in the system, (lambda) is throughput per unit time, and (W) is average time in the system. For a resource:

$$
\widehat{O}_{h,r,t}=\sum_c \widehat{\lambda}_{h,c,t}\,\widehat{W}_{h,c,r,t}.
$$

The model must reconcile this estimate with observed or replayed occupancy, admissions, discharges, and transfers. A mismatch is a diagnostic signal; it must not be hidden by a downstream optimizer.

## 6. Queueing and service capacity

### 6.1 M/M/s approximation

For (s) identical servers, arrival rate (lambda), and service rate (mu), utilization is:

$$
\rho=\frac{\lambda}{s\mu}.
$$

The Erlang-C waiting probability is:

$$
P_W=\frac{\frac{a^s}{s!}\frac{1}{1-\rho}}
{\sum_{k=0}^{s-1}\frac{a^k}{k!}+\frac{a^s}{s!}\frac{1}{1-\rho}},
\qquad a=\frac{\lambda}{\mu}.
$$

The expected queue wait is:

$$
W_q=\frac{P_W}{s\mu-\lambda}.
$$

These formulas require (ho<1) and idealized assumptions. MINCO uses them as an interpretable approximation and complements them with discrete-event simulation.

### 6.2 Kingman's variability approximation

For a G/G/1-style approximation:

$$
W_q \approx \frac{\rho}{1-\rho}\cdot
\frac{c_a^2+c_s^2}{2}\cdot\mathbb{E}[S]
$$

where (c_a) and (c_s) are coefficients of variation for interarrival and service times and (mathbb{E}[S]) is mean service time. The equation makes variability visible: two systems with the same mean load can have different waits when arrival or service variability differs.

### 6.3 Capacity and overflow

For a staffed capacity (B_{h,r,t}), an operational excess measure can be defined as:

$$
X_{h,r,t}=\max(0,O_{h,r,t}-B_{h,r,t}).
$$

Unsafe excess, overflow, blocked arrivals, and rejected elective demand should remain separate metrics. Combining them into one opaque score prevents operators from understanding the tradeoff.

## 7. Correlated stochastic scenarios

The optimizer should not see one deterministic forecast. It should see a scenario set:

$$
\mathcal{S}=\{(A_s,O_s,W_s,\text{transfers}_s):s=1,\ldots,S\}.
$$

Each scenario should preserve plausible dependence among:

- facilities exposed to the same network pressure;
- ED arrivals and ICU escalation;
- length of stay and acuity;
- transfer demand and receiving capacity;
- staffing availability and effective beds.

Independent sampling of each variable can dramatically understate tail risk. A practical generator can combine:

1. a shared latent regime;
2. facility random effects;
3. cohort-specific conditional models;
4. correlated residual draws or a copula;
5. operational constraints and conservation checks.

The generated scenario must be checked for nonnegative counts, feasible timestamp order, conservation of arrivals and departures, and plausible marginal distributions. The current common-random-number design reuses random streams for comparable plans so that differences are less contaminated by Monte Carlo noise.

## 8. Two-stage stochastic capacity optimization

### 8.1 Decision structure

First-stage decisions are selected before the realized scenario is known:

- activate surge beds;
- activate flex staff blocks;
- reserve or expand transfer capacity;
- defer or protect elective capacity;
- configure network-level policy.

Second-stage decisions adapt to scenario (s):

- transfers and diversions;
- recourse staffing;
- overflow or service substitutions;
- unmet-demand penalties.

Let (x) be first-stage decisions and (y_s) be recourse decisions. The generic problem is:

$$
\min_{x,y_s}\; C_0(x)+\mathbb{E}_{s}[C_s(x,y_s)]
$$

subject to:

$$
x\in X,
\qquad y_s\in Y_s(x),\quad s\in S.
$$

### 8.2 Capacity and conservation constraints

For resource (r) at facility (h):

$$
O_{h,r,t,s} \le B_{h,r,t}+\Delta B_{h,r,t}(x,y_s)+u_{h,r,t,s}
$$

where (u) is explicitly penalized unsafe excess or overflow. Flow conservation for a facility/resource can be represented as:

$$
O_{h,r,t+1,s}=O_{h,r,t,s}+A_{h,r,t,s}+\operatorname{in}_{h,r,t,s}
-\operatorname{out}_{h,r,t,s}-D_{h,r,t,s}.
$$

For a transfer lane ((h,k)):

$$
F_{h,k,t,s}\le \overline{F}_{h,k,t}+\Delta F_{h,k,t}(x,y_s)
$$

and receiving capacity must also be enforced. A network graph edge alone does not imply unlimited transfer volume.

Binary activation and integer staffing/beds make the problem mixed-integer. If a nonlinear risk or queue approximation is used, it should be linearized, bounded, or moved into the simulation layer with its approximation documented.

### 8.3 CVaR tail-risk objective

For scenario loss (L_s(x,y_s)), Conditional Value at Risk at level (alpha) is:

$$
\operatorname{CVaR}_{\alpha}(L)=
\min_{\eta}\left[\eta+\frac{1}{1-\alpha}
\mathbb{E}\left[(L-\eta)_+\right]\right].
$$

For a finite scenario set, introduce (z_s\ge0):

$$
z_s\ge L_s-\eta,
$$

then minimize:

$$
C_0(x)+\sum_s p_s L_s(x,y_s)+gamma\left(\eta+
\frac{1}{1-\alpha}\sum_s p_s z_s\right).
$$

Here (gamma) controls the tradeoff between expected cost and tail risk. A higher (gamma) or higher (alpha) generally protects more strongly against severe scenarios, but may increase ordinary operating cost. The user interface must show the selected (alpha), (gamma), scenario count, and objective components.

### 8.4 Solver behavior and fallback

MINCO’s intended primary path is Julia/JuMP/Gurobi, with a SciPy/HiGHS verification path for small instances and environments where the primary project is not available. A solver status of `OPTIMAL`, `TIME_LIMIT`, `INFEASIBLE`, or fallback must be surfaced distinctly. A fallback is not mathematically interchangeable with the primary solver unless an oracle comparison has established the relevant equivalence for the instance.

## 9. Simulation and uncertainty quantification

### 9.1 Discrete-event simulation

The discrete-event layer samples arrivals, service, escalation, discharge, and transfer events on a timeline. It evaluates consequences that are difficult to encode in a compact MILP, including queue buildup, time-dependent congestion, and event ordering.

For (R) replications, an estimated metric mean is:

$$
\bar{M}=\frac{1}{R}\sum_{r=1}^{R}M_r.
$$

An approximate standard error is:

$$
\operatorname{SE}(\bar{M})=\frac{s_M}{\sqrt{R}}.
$$

The number of replications should be chosen from a precision requirement, not a UI default. Tail metrics need more replications than means.

### 9.2 Common random numbers

When comparing a baseline plan and a counterfactual, use the same random seeds and latent shocks for both plans where scientifically appropriate:

$$
\widehat{\Delta}=\frac{1}{R}\sum_{r=1}^{R}
\left(M_r^{\text{counterfactual}}-M_r^{\text{baseline}}\right).
$$

This paired estimate often has lower variance than subtracting two independently noisy means. The seed policy must be recorded in the run manifest so results can be replayed.

### 9.3 Stress-test design

Stress tests should include mode shifts, facility closure, delayed events, transfer-lane failure, staffing reduction, missing data, extreme arrivals, and optimization timeout. A stress test is useful only when the system explains whether failure came from data, prediction, constraints, solver status, or simulation.

## 10. ML foundations

### 10.1 Forecast target and information set

An operational forecast must define an information set (mathcal{F}_t) containing only information available at prediction time:

$$
\widehat{Y}_{t+h}=f(\mathcal{F}_t).
$$

For next-day arrivals, (mathcal{F}_t) may include lagged arrivals, day-of-week, season, recent occupancy, recent discharges, facility identity, and approved aggregate surveillance variables. It must not include future-corrected records or a label-derived feature.

The target, horizon, cutoff timestamp, missingness behavior, and censoring rule belong in the model manifest.

### 10.2 Quantile demand forecasting

Instead of predicting only a conditional mean, estimate quantiles:

$$
\widehat{q}_{\tau}(Y\mid X)=f_{\tau}(X),\qquad \tau\in\{0.1,0.5,0.9\}.
$$

Quantile loss, or pinball loss, is:

$$
\ell_{\tau}(y,q)=
\begin{cases}
\tau(y-q), & y\ge q,\\
(\tau-1)(y-q), & y<q.
\end{cases}
$$

MINCO’s quantile-gradient-boosting path produces forecast intervals that can parameterize scenarios. It must be compared to seasonal-naive and other transparent baselines. Interval coverage and width should be measured separately:

$$
\operatorname{Coverage}=\frac{1}{n}\sum_{i=1}^n
\mathbf{1}\{y_i\in[L_i,U_i]\}.
$$

### 10.3 Discrete-time discharge hazard

Length of stay is represented as a sequence of conditional discharge probabilities rather than one unconstrained regression target. For patient/cohort (i) at interval (k):

$$
h_{i,k}=P(T_i=k\mid T_i\ge k,X_i).
$$

The survival probability through interval (k) is:

$$
S_i(k)=\prod_{j=1}^{k}(1-h_{i,j}),
$$

and the probability of discharge at (k) is:

$$
P(T_i=k)=S_i(k-1)h_{i,k}.
$$

This representation naturally handles right-censoring when the observation ends before discharge. Calibration and survival-curve behavior matter more than a single classification score.

### 10.4 ICU-escalation risk

For a defined horizon (H), the escalation model estimates:

$$
p_i=P(\text{ICU escalation within }H\mid X_i).
$$

Features may include age band, acuity, oxygen-support proxy, ED pressure, recent trajectory, and facility context, subject to a data-use and clinical review. The probability is used to parameterize capacity scenarios; it is not a diagnosis.

The current implementation uses calibrated gradient boosting. A calibrated probability should satisfy:

$$
P(Y=1\mid \widehat{p}\approx p)\approx p.
$$

Calibration can be assessed with reliability curves, expected calibration error, and Brier score:

$$
\operatorname{Brier}=\frac{1}{n}\sum_{i=1}^{n}(p_i-y_i)^2.
$$

### 10.5 Temporal, facility-aware validation

The minimum defensible split is chronological:

```text
training period → tuning/validation period → future holdout period
```

For network use, add facility-aware holdouts or grouped evaluation. Report:

- ROC-AUC and PR-AUC for ranking;
- Brier score and calibration curves for probabilities;
- precision, recall, alert volume, and false-alert burden at policy thresholds;
- performance by facility size, service, mode, season, acuity, and data-quality tier;
- abstention and missingness behavior;
- confidence intervals through bootstrap or repeated temporal folds.

The bundled research acceptance gate uses a temporal holdout and checks that the model’s Brier score improves on a constant-probability baseline. That is a useful regression gate, not evidence of clinical benefit.

### 10.6 Leakage controls

Common leakage paths include future occupancy, post-outcome coding, corrected timestamps, future discharge status, replication identifiers, and features generated after the prediction cutoff. Controls should include:

- point-in-time feature construction;
- immutable cutoff timestamps;
- feature availability tests;
- cohort separation between training and validation;
- duplicate and encounter-overlap checks;
- a leakage review recorded in the model manifest.

## 11. AI layer and human-language assistance

MINCO uses “AI” in two different senses that must remain separate:

1. **Computational AI/ML** — statistical forecasting, hazard estimation, calibrated risk, regime inference, scenario generation, and uncertainty estimation.
2. **Optional generative assistance** — explanation, retrieval, summarization, and question answering over already governed artifacts.

The generative layer must not:

- invent data or evidence;
- alter solver constraints or objective weights;
- convert a recommendation into an authorized action;
- hide uncertainty, stale data, failed checks, or abstention;
- receive raw PHI unless a separately approved architecture and vendor agreement permits it.

Every generated explanation should cite the run ID, model/version, scenario set, evidence artifacts, and relevant assumptions. Prompt and retrieval versions belong in the audit manifest. The deterministic mathematical path must remain usable when the language model is unavailable.

## 12. AI → IE → OR coupling contract

The coupling should be explicit and typed:

```text
forecast model
  → estimate + interval + calibration metadata
risk model
  → probability + calibration metadata + abstention flag
regime model
  → mode probabilities or selected research mode
scenario generator
  → scenarios + dependence metadata + seed manifest
IE layer
  → expected resource-days + queue metrics + conservation diagnostics
OR layer
  → feasible first-stage/recourse plan + objective decomposition
simulation
  → stress metrics + confidence intervals + failure diagnostics
review layer
  → human disposition + rationale + immutable audit event
```

If any upstream component is stale, outside scope, uncalibrated, or failed, the downstream component should receive an explicit status and may abstain. It should not silently substitute a point estimate.

## 13. Fairness, robustness, and model risk

Hospital capacity tools can create uneven burdens even when they do not make individual clinical decisions. Evaluate:

- service access and elective rejection by facility and population proxy;
- transfer burden and travel burden across facilities;
- alert burden across sites and shifts;
- calibration and missingness across relevant subgroups;
- sensitivity to objective weights and scenario assumptions;
- worst-case behavior under data delay and facility closure.

Where protected characteristics are not appropriate model features, they may still be needed in a restricted validation set to detect disparate performance. The governance process must define who may access that evaluation data.

## 14. Implementation map

| Foundation | Current implementation area |
|---|---|
| MMPP / regime inference | `src/stochastic/mmpp.py`, `src/ai/calibrated_regime_forecast.py` |
| CTMC patient flow | `src/stochastic/ctmc.py`, `src/ie/healthcare_flow.py` |
| Conservation and staffed capacity | `src/decision_math/flow_conservation.py` |
| Queueing approximations | `src/ie/healthcare_flow.py` |
| Scenario generation | `src/stochastic/decision_scenarios.py`, `src/scenarios/operating_modes.py` |
| Monte Carlo / DES | `src/stochastic/monte_carlo.py`, `src/benchmarks/stochastic_capacity.py` |
| Quantile demand forecast | `src/ai/demand_forecast_quantile.py` |
| Discharge hazard | `src/ml/discharge_hazard.py` |
| ICU escalation | `src/ml/icu_escalation.py` |
| FLOW-CVaR model | `src/decision_math/flow_cvar.py`, `src/decision_math/flow_cvar_decision.py` |
| Progressive hedging / rolling horizon | `src/decision_math/progressive_hedging.py`, `src/decision_math/rolling_horizon.py` |
| Independent optimization oracle | `src/decision_math/stochastic_milp_oracle.py` |
| Scaled replay and public calibration | `src/scenarios/scaled_replay.py`, `src/scenarios/public_reference.py` |
| Runtime contract | `src/runtime/contracts.py`, `src/runtime/workstation_service.py` |
| Evidence and governance | `src/runtime/evidence_tools.py`, `src/services/release_readiness_service.py` |

## 15. Required validation gates

### Mathematical gate

- unit tests for generator construction, fundamental matrix, Little’s Law, conservation, queueing, and CVaR linearization;
- independent oracle on small instances;
- monotonicity and infeasibility tests;
- solver status, tolerance, timeout, and fallback tests;
- deterministic rerun with fixed seeds.

### ML gate

- point-in-time feature test;
- chronological and facility-aware holdout;
- baseline comparison;
- calibration and interval coverage;
- subgroup/facility/mode stratification;
- drift and abstention tests;
- signed model/data/dependency manifest.

### Integrated gate

- forecast-to-scenario lineage;
- scenario-to-optimizer lineage;
- optimizer-to-simulation comparison;
- decision packet containing assumptions, objective components, uncertainty, solver status, and review history;
- deliberate failures for stale data, missing facilities, invalid events, solver timeout, and audit tampering.

## 16. Limitations and next mathematical upgrades

The current research implementation is intentionally transparent but simplified. The next serious upgrades are:

1. estimate hierarchical facility effects and cross-facility dependence from approved time-stamped data;
2. add Bayesian or conformal uncertainty for small facilities and sparse services;
3. model time-varying transition rates and competing risks for discharge, transfer, and escalation;
4. add explicit staffing skill-mix, shift, fatigue, and labor-rule constraints;
5. incorporate robust or distributionally robust optimization with an uncertainty-set specification;
6. validate simulation with calibration curves, queue distributions, and historical event traces;
7. quantify decision utility, not just predictive accuracy;
8. provide a controlled causal or quasi-experimental evaluation for any claim that MINCO improves operations;
9. separate research-mode parameters from locally approved operational policies;
10. keep all model, solver, data, prompt, and configuration changes versioned and auditable.

## 17. Plain-language summary

The math predicts how people and demand move through constrained hospital resources. The ML estimates uncertain inputs such as arrivals, discharge readiness, and escalation probability. The optimizer chooses a feasible plan while balancing average cost and severe-tail risk. Simulation asks what happens when reality is noisy and events arrive in an inconvenient order. The AI explanation layer helps a human understand the result, but it is not the authority that makes or executes the decision.
