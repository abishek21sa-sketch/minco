# MINCO Phase 1 Mathematical Methods

## A. Artificial Intelligence

### Poisson Hidden Markov demand-regime model

- **Task:** infer latent operational pressure regime from hourly arrival counts.
- **Model:** Poisson-emission Hidden Markov Model estimated by Baum-Welch EM.
- **Emission:** `Y_t | Z_t=k ~ Poisson(lambda_k)`.
- **Transition:** `P(Z_t=j | Z_{t-1}=i)=A_ij`.
- **Validation:** known synthetic generative regimes; fitted rate error, posterior regime recovery and EM log-likelihood progression.
- **Decision use:** supplies stochastic regime dynamics and demand rates to scenario generation.
- **Implementation:** `src/stochastic/mmpp.py`.

### Discharge hazard model

- **Task:** estimate bed-release probability by horizon.
- **Model:** discrete-time logistic hazard model on person-period data.
- **Output:** cumulative `P(discharge <= h | X)`.
- **Validation:** held-out synthetic patient stays; Brier score against constant probability baseline.
- **Decision use:** release probability reduces future ICU census in Monte Carlo scenario generation.
- **Implementation:** `src/ml/discharge_hazard.py`.

### ICU escalation model

- **Task:** estimate operational probability of ICU escalation within 24 hours.
- **Model:** calibrated Gradient Boosting classifier.
- **Metrics:** ROC-AUC, PR-AUC and Brier score.
- **Decision use:** escalation probabilities parameterize stochastic ICU-census scenarios consumed by the MILP.
- **Boundary:** operational synthetic research model, not diagnosis.
- **Implementation:** `src/ml/icu_escalation.py`.

## B. Industrial Engineering

### Continuous-time Markov patient flow

`P(t) = exp(Qt)` where `Q` is the CTMC generator.

For transient generator `T`, expected state residence time is obtained from:

`N = (-T)^(-1)`.

Units are hours and expected hours/patient. Implemented in `src/stochastic/ctmc.py`.

### Patient-flow conservation

`C_(u,t+1) = C_(u,t) + A_(u,t) + T_in - D_(u,t) - T_out`.

Residual must be zero within numerical tolerance. Implemented in `src/decision_math/flow_conservation.py`.

### Queueing

For M/M/s:

`rho = lambda / (s * mu)`.

Erlang-C supplies probability of waiting and expected queue wait. Kingman's GI/G/1 approximation introduces arrival/service variability through squared coefficients of variation. Little's Law verifies `L = lambda W` relationships. Implemented in `src/queueing/hospital_queueing.py`.

### Staffed capacity

Available physical beds are capped by staffing capability:

`staffed_beds = min(physical_beds, available_nurse_hours / nurse_hours_per_bed)`.

Implemented and unit-tested in `src/decision_math/flow_conservation.py`.

## C. Operations Research

### Two-stage stochastic hospital-network MILP

**First stage**

- `y[h,t] ∈ {0,1}`: surge-pod activation.
- `f[h,t] ∈ Z+`: flex-staff blocks.

**Scenario recourse**

- `d[w,h,t] ∈ Z+`: controllable demand deferred.
- `x[w,i,j,t] ∈ Z+`: inter-hospital transfer count.
- `u[w,h,t] >= 0`: unsafe-capacity slack.
- `b[w,h,t] >= 0`: boarding/physical-capacity slack.

The objective combines activation cost, expected recourse cost and tail risk:

`min C_first + E[L_w] + beta * CVaR_alpha(L_w)`.

CVaR linearization uses `eta` and excess variables `xi_w`:

`xi_w >= L_w - eta`, `xi_w >= 0`.

Constraints include surge/flex limits, deferral limits, lane transfer capacities, safe-capacity pressure and boarding pressure.

**Primary implementation:** `julia/src/extensive_form.jl` using JuMP/Gurobi.

**Independent oracle:** `src/decision_math/stochastic_milp_oracle.py` using SciPy/HiGHS. A one-hospital instance is also solved by exact first-stage enumeration.

### Progressive Hedging

Scenario MILPs/MIQPs are decomposed and first-stage nonanticipativity is enforced iteratively with augmented penalties. Phase 1 applies Progressive Hedging to the risk-neutral expected-recourse formulation; the CVaR formulation is solved as an extensive form. This boundary is explicit rather than implying a decomposed CVaR algorithm that is not implemented. Primary implementation is `julia/src/progressive_hedging.jl`; a tiny independent reference is `src/decision_math/progressive_hedging.py`.

### Rolling horizon

Only the first-period first-stage action is committed before state/demand scenarios are refreshed and the model is solved again. Implemented in Julia and independently checked in Python.

## Integrated decision chain

Phase 1 explicitly validates:

```text
MMPP regime inference
+ discharge hazard probability
+ ICU escalation probability
        ↓
stochastic ICU census tensor
        ↓
CVaR stochastic MILP
        ↓
integer surge / flex / transfer decisions
        ↓
independent feasibility validation
```
