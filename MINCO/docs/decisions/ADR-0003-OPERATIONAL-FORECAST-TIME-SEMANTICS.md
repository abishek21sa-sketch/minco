# ADR-0003 — Supersede Replication-Index Forecasting for Operational Claims

**Status:** Accepted — Finalization Phase 1

## Context

The earlier forecast dataset treated ordered Monte Carlo replications as a `time_index` and formed next-step targets by shifting rows within scenario/policy groups. Independent stochastic replications do not constitute chronological hospital operations.

## Decision

The earlier models remain historical research artifacts only. Operational forecast claims now require a genuine calendar-time axis and features available before the forecast date. Finalization Phase 1 uses generated synthetic historical replay, chronological train/calibration/test partitions, a seasonal-naive baseline, quantile Gradient Boosting, and conformalized prediction intervals.

## Consequences

No real-hospital accuracy claim is permitted. The new model is **VALIDATED ON SYNTHETIC HISTORICAL REPLAY** only.
