"""Statistical qualification for analytical claims derived from replication data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config.paths import RESULTS_DIR
from src.validation.run_manifest import new_run_id, write_run_manifest


def paired_policy_comparison(
    frame: pd.DataFrame,
    *,
    policy_a: str,
    policy_b: str,
    metric: str,
    group_columns: Iterable[str] = ("scenario",),
    lower_is_better: bool = True,
    bootstrap_iterations: int = 5000,
    seed: int = 20260730,
) -> pd.DataFrame:
    """Compare two policies using paired replications and a seeded bootstrap CI."""
    groups = list(group_columns)
    required = set(groups + ["replication", "policy_name", metric])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Claim comparison is missing columns: {missing}")
    if bootstrap_iterations < 200:
        raise ValueError("At least 200 bootstrap iterations are required")

    selected = frame[frame["policy_name"].isin([policy_a, policy_b])].copy()
    records: list[dict[str, Any]] = []
    grouped = selected.groupby(groups, dropna=False) if groups else [((), selected)]
    rng = np.random.default_rng(seed)

    for group_key, subset in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        wide = subset.pivot_table(
            index="replication", columns="policy_name", values=metric, aggfunc="first"
        )
        if policy_a not in wide.columns or policy_b not in wide.columns:
            continue
        paired = wide[[policy_a, policy_b]].dropna()
        if paired.empty:
            continue
        differences = paired[policy_a].to_numpy(float) - paired[policy_b].to_numpy(float)
        boot = np.empty(bootstrap_iterations, dtype=float)
        n = len(differences)
        for idx in range(bootstrap_iterations):
            sample = rng.choice(differences, size=n, replace=True)
            boot[idx] = float(sample.mean())
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        mean_a = float(paired[policy_a].mean())
        mean_b = float(paired[policy_b].mean())
        mean_difference = mean_a - mean_b
        advantage = mean_difference < 0 if lower_is_better else mean_difference > 0
        statistically_supported = ci_high < 0 if lower_is_better else ci_low > 0
        if advantage and statistically_supported:
            classification = "validated_advantage"
        elif advantage:
            classification = "descriptive_advantage"
        else:
            classification = "no_observed_advantage"
        denominator = abs(mean_b)
        relative_improvement = (
            ((mean_b - mean_a) / denominator * 100.0)
            if lower_is_better and denominator > 0
            else ((mean_a - mean_b) / denominator * 100.0 if denominator > 0 else np.nan)
        )
        record: dict[str, Any] = {
            "policy_a": policy_a,
            "policy_b": policy_b,
            "metric": metric,
            "lower_is_better": lower_is_better,
            "n_paired_replications": n,
            "policy_a_mean": mean_a,
            "policy_b_mean": mean_b,
            "mean_difference_a_minus_b": mean_difference,
            "bootstrap_ci95_low": float(ci_low),
            "bootstrap_ci95_high": float(ci_high),
            "relative_improvement_percent": float(relative_improvement),
            "claim_classification": classification,
        }
        record.update(dict(zip(groups, key_values)))
        records.append(record)
    return pd.DataFrame(records)


def run_claim_governance_validation(
    *,
    results_dir: Path = RESULTS_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Qualify headline policy claims from stored paired replication evidence."""
    output_dir = Path(output_dir or (RESULTS_DIR / "validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = results_dir / "policy_comparison_baseline_replications.csv"
    robust_path = results_dir / "tables" / "robust_vs_nominal_replications.csv"
    if not baseline_path.exists() or not robust_path.exists():
        missing = [str(path) for path in (baseline_path, robust_path) if not path.exists()]
        raise FileNotFoundError(f"Missing claim-governance evidence: {missing}")

    baseline = pd.read_csv(baseline_path)
    robust = pd.read_csv(robust_path)
    comparisons = [
        paired_policy_comparison(
            baseline,
            policy_a="optimized_network",
            policy_b="no_control",
            metric="total_unsafe_excess",
        ),
        paired_policy_comparison(
            baseline,
            policy_a="optimized_network",
            policy_b="no_control",
            metric="max_utilization_ratio",
        ),
        paired_policy_comparison(
            robust,
            policy_a="robust_optimized_network",
            policy_b="optimized_network",
            metric="total_unsafe_excess",
        ),
    ]
    table = pd.concat(comparisons, ignore_index=True)
    output_path = output_dir / "claim_governance.csv"
    summary_path = output_dir / "claim_governance_summary.json"
    table.to_csv(output_path, index=False)

    counts = table["claim_classification"].value_counts().to_dict()
    payload = {
        "status": "passed" if not table.empty else "failed",
        "method": "paired_seeded_bootstrap_mean_difference",
        "comparison_rows": int(len(table)),
        "classification_counts": {str(k): int(v) for k, v in counts.items()},
        "approved_interpretation": {
            "validated_advantage": "Mean advantage with a paired 95% bootstrap interval excluding zero.",
            "descriptive_advantage": "Observed mean advantage; uncertainty interval includes zero.",
            "no_observed_advantage": "No lower mean was observed for the candidate policy.",
        },
        "output_path": str(output_path),
    }
    run_id = new_run_id("claim_governance")
    payload["run_id"] = run_id
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest_path, manifest_sha = write_run_manifest(
        run_id=run_id,
        run_type="claim_governance",
        parameters={"bootstrap_iterations": 5000, "seed": 20260730},
        input_paths=[baseline_path, robust_path],
        output_paths=[output_path],
        metrics={"comparison_rows": int(len(table)), **payload["classification_counts"]},
        notes=[
            "Numerical percentage reductions and inferential superiority are reported separately.",
            "Stored experiments use the synthetic reference network and do not establish real-hospital impact.",
        ],
    )
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_sha256"] = manifest_sha
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if table.empty:
        raise AssertionError("Claim governance produced no comparisons")
    return payload
