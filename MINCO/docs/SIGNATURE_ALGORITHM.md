# FLOW-CVaR Signature Algorithm

Canonical executable module: `src/minco4x/signature_algorithm.py`.

The governed reference propagates a row-stochastic Markov patient-flow state across H1, H2 and EXIT, adds governed new arrivals, generates reference/rare-surge demand, and solves a two-hospital stochastic capacity MILP with surge beds, flex-staff-equivalent capacity, transfers, diversion/deferral recourse and CVaR.

See `docs/FLOW_CVAR.md` and `docs/RESEARCH_VALIDATION_FLOW_CVAR.md`.
