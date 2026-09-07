# Level 2 WP2 — Simulation, Uncertainty, and Optimization Robustness

## Purpose

This work package validates the internal behavior of MINCO's preserved synthetic analytical core. It does not claim real-hospital calibration.

## Evidence gates

1. Every simulated patient remains accounted for across ED, ICU, Ward, Discharged, and Dead states.
2. Monte Carlo state means converge to the deterministic Markov expectation within declared tolerances.
3. The Poisson arrival generator reproduces its configured synthetic mean and variance.
4. Policy claims are classified using paired replication evidence and seeded bootstrap confidence intervals.
5. Capacity relaxation, demand stress, and transfer restriction satisfy expected optimization monotonicity.
6. The legacy `robust_optimized_network` label is explicitly classified as uncertainty-adjusted nominal optimization, not formal robust optimization.

## Public claim boundary

Approved: "MINCO's stochastic state-transition simulator was internally verified against its known synthetic Markov and Poisson assumptions."

Not approved: "The digital twin is calibrated to real hospital operations" or "MINCO implements formal robust optimization."
