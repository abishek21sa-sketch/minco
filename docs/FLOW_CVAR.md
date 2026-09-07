# FLOW-CVaR — Patient-Flow Tail-Risk Capacity Planning

FLOW-CVaR is MINCO's compact governed decision layer over its two-stage stochastic capacity model. First-stage decisions activate surge beds and flex-staff capacity. Scenario recourse includes elective deferral, transfers, unsafe-capacity excess and boarding. Scenario probabilities represent uncertain patient demand.

The objective minimizes first-stage capacity cost plus expected recourse loss and a Rockafellar-Uryasev CVaR penalty. The patient-flow semantics remain tied to MINCO's capacity/transfer model; the layer does not use an LLM as an optimizer.

## Evidence boundary

The included HiGHS implementation is an independent small-instance validation oracle for the primary Julia/JuMP/Gurobi architecture. Evidence demonstrates mathematical feasibility, solver/oracle consistency and tail-risk behavior on controlled scenarios. It does **not** establish clinical effectiveness, calibrated hospital demand, or patient-outcome guarantees.
