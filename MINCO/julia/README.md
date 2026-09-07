# MINCO Julia/JuMP decision engine

This is the primary Phase-1 Operations Research runtime. It uses JuMP with
Gurobi for a two-stage stochastic hospital-network MILP, CVaR tail-risk,
Progressive Hedging scenario decomposition, and rolling-horizon reoptimization.

## Windows setup

```powershell
julia julia/setup.jl
```

Gurobi licensing is external to this repository. No license or WLS credential
must be committed.
