from src.ml.icu_escalation import (
    ICUEscalationModel,
    evaluate_escalation_model,
    synthetic_escalation_dataset,
)


def test_calibrated_icu_escalation_model_beats_constant_probability_on_synthetic_holdout():
    train = synthetic_escalation_dataset(2200, seed=10)
    test = synthetic_escalation_dataset(800, seed=11)
    model = ICUEscalationModel().fit(train)
    metrics = evaluate_escalation_model(model, test)
    assert metrics["roc_auc"] >= 0.78
    assert metrics["brier_score"] < metrics["constant_brier_score"]
    assert metrics["pr_auc"] > metrics["prevalence"]
