# Level 2 WP1 Local Validation Checklist

- [ ] Full pytest suite passes with no skipped Gurobi tests.
- [ ] `/readiness` reports Gurobi package and license verified.
- [ ] Forecast baseline validation writes CSV, summary JSON, and run manifest.
- [ ] A new `/what-if` response contains `manifest.path` and `manifest.sha256`.
- [ ] `/audit-log` returns the same manifest reference for the run.
- [ ] Re-running the base optimizer produces an identical objective within tolerance.
- [ ] No public forecasting claim is updated until candidate models are evaluated on these splits.
