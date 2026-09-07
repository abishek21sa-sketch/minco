"""Acceptance harness for the scaled MINCO synthetic research twin."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.benchmarks.stochastic_capacity import make_network_capacity_benchmark
from src.config.paths import RESULTS_DIR
from src.ml.icu_escalation import ICUEscalationModel, evaluate_escalation_model
from src.scenarios.operating_modes import MODE_CATALOG
from src.scenarios.scaled_replay import build_scaled_event_replay


def main() -> int:
    public_csv = Path("data/public_reference/cms_hospital_general_information.csv")
    normal = build_scaled_event_replay(
        n_hospitals=120,
        horizon_days=30,
        minimum_events=100_000,
        operating_mode="normal",
        public_reference_csv=public_csv,
    )
    counts = Counter(event.event_type.value for event in normal)
    facility_ids = {event.hospital_id for event in normal}
    arrival_frame = pd.DataFrame(
        [
            {
                **{
                    key: event.metadata[key]
                    for key in (
                        "age",
                        "acuity_score",
                        "shock_index_proxy",
                        "oxygen_need_proxy",
                        "arrival_mode",
                        "icu_escalation_24h",
                    )
                },
                "cohort": event.patient_pathway_or_cohort,
            }
            for event in normal
            if event.event_type.value == "arrival"
        ]
    )
    split = int(len(arrival_frame) * 0.75)
    ml_model = ICUEscalationModel().fit(arrival_frame.iloc[:split].reset_index(drop=True))
    ml_metrics = evaluate_escalation_model(
        ml_model, arrival_frame.iloc[split:].reset_index(drop=True)
    )
    if (
        ml_metrics["roc_auc"] < 0.65
        or ml_metrics["brier_score"] >= ml_metrics["constant_brier_score"]
    ):
        raise RuntimeError(f"Scaled ML calibration gate failed: {ml_metrics}")
    mode_smoke_counts = {
        mode: len(
            build_scaled_event_replay(
                n_hospitals=24,
                horizon_days=10,
                minimum_events=5_000,
                operating_mode=mode,
                public_reference_csv=public_csv,
            )
        )
        for mode in MODE_CATALOG
    }
    benchmark = make_network_capacity_benchmark(
        n_hospitals=24,
        n_periods=4,
        n_scenarios=8,
        operating_mode="covid_like",
    )
    report = {
        "status": "PASS",
        "event_count": len(normal),
        "facility_count": len(facility_ids),
        "event_type_counts": dict(counts),
        "source_modes": sorted({event.source_mode.value for event in normal}),
        "public_reference_snapshot_present": public_csv.exists(),
        "mode_smoke_event_counts": mode_smoke_counts,
        "mode_catalog": sorted(MODE_CATALOG),
        "optimizer_scale_probe": {
            "hospitals": len(benchmark.hospitals),
            "periods": len(benchmark.periods),
            "scenarios": len(benchmark.scenario_probabilities),
            "demand_shape": list(benchmark.demand.shape),
        },
        "ml_validation": {
            "model": "calibrated ICU escalation classifier",
            "training_rows": split,
            "temporal_holdout_rows": len(arrival_frame) - split,
            **ml_metrics,
            "gate": "ROC-AUC >= 0.65 and Brier score better than prevalence baseline",
        },
        "claim_boundary": (
            "Synthetic research/replay validation only. Public CMS facility metadata is calibration context; "
            "no patient-level data, observed occupancy, clinical outcome, or autonomous execution claim is made."
        ),
    }
    output = (
        RESULTS_DIR
        / "validation"
        / "scaled_research_twin"
        / "scaled_research_twin_acceptance_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "report_path": str(output),
                "event_count": len(normal),
                "facility_count": len(facility_ids),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
