# Level 2 WP3 — Predictive Model Validation and Governance

## Objective

Replace historical random-split model claims with time-respecting,
scenario-held-out evidence and explicit model approval decisions.

## Governing rules

- No random row split.
- No future outcome columns in the feature matrix.
- Replication identifiers are excluded.
- Every candidate is compared with an operationally meaningful baseline.
- Classification probability quality is measured with log loss, Brier score,
  and expected calibration error.
- A task is rejected when its target is deterministically constructed from
  current-state labels.
- Approval applies only to the bundled synthetic Meridian reference case.

## Tasks evaluated

1. Next-period unsafe excess regression.
2. Next-period blocked-arrival regression.
3. Next-period critical-utilization classification.
4. Next-regime classification.

The next-regime task is expected to be rejected when `current_regime` or
`current_regime_code` deterministically constructs the target. A perfect score
under that condition is not evidence of useful forecasting.

## Outputs

- `results/validation/predictive_model_validation.csv`
- `results/validation/predictive_feature_audit.csv`
- `results/validation/predictive_model_governance_summary.json`
- `results/model_registry/predictive_model_registry.csv`
- `results/model_registry/models/*.joblib`
- `results/run_manifests/predictive_governance_*.json`
- `results/validation/level2_wp3_validation_report.json`

## Acceptance command

```powershell
python -m src.system.run_level2_wp3_validation
```

The command must return `"status": "passed"`. This means the governance process
ran reproducibly; it does not mean that every model was approved.
