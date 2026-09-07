"""Operational-state reconstruction service for the reference benchmark."""
from __future__ import annotations

from pathlib import Path

from src.config.loader import load_healthcare_instance
from src.config.paths import DATA_DIR
from src.contracts.common import ExecutionContext, ServiceMetadata, new_correlation_id, utc_now
from src.contracts.operational_state import OperationalStateResponse, OperationalStateSnapshot
from src.validation.run_manifest import describe_path


class OperationalStateService:
    """Reconstruct a typed snapshot from the current operational input tables."""

    service_name = "operational_state_service"

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

    def get_reference_snapshot(
        self,
        context: ExecutionContext | None = None,
    ) -> OperationalStateResponse:
        context = context or ExecutionContext(correlation_id=new_correlation_id(), source="service")
        instance = load_healthcare_instance(self.data_dir)

        hospitals_df = instance.hospitals.df
        arrivals_df = instance.arrivals.df
        capacities_df = instance.capacities.df
        thresholds_df = instance.safe_thresholds.df
        lanes_df = instance.transfer_lanes.df

        hospital_ids = sorted(hospitals_df["hospital_id"].astype(str).unique().tolist())
        cohort_ids = sorted(arrivals_df["cohort"].astype(str).unique().tolist())
        capacity_by_resource = {
            str(resource): float(value)
            for resource, value in capacities_df.groupby("resource")["base_capacity"].sum().items()
        }
        enabled_lanes = lanes_df[lanes_df["allowed"].astype(float) > 0].copy()
        transfer_count = int(len(enabled_lanes))
        total_transfer_capacity = float(enabled_lanes["transfer_capacity"].sum())

        base_fingerprint = describe_path(self.data_dir / "base_instance")
        transition_fingerprint = describe_path(self.data_dir / "transitions")
        fingerprints = {
            "base_instance": str(base_fingerprint.get("sha256", "")),
            "transitions": str(transition_fingerprint.get("sha256", "")),
        }

        snapshot = OperationalStateSnapshot(
            snapshot_id=f"state_{new_correlation_id().removeprefix('corr_')[:16]}",
            observed_at=utc_now(),
            hospital_count=len(hospital_ids),
            hospital_ids=hospital_ids,
            cohort_count=len(cohort_ids),
            cohort_ids=cohort_ids,
            horizon_days=int(arrivals_df["day"].nunique()),
            arrival_rows=int(len(arrivals_df)),
            total_expected_arrivals=float(arrivals_df["arrivals"].sum()),
            capacity_by_resource=capacity_by_resource,
            safe_utilization_min=float(thresholds_df["safe_utilization"].min()),
            safe_utilization_max=float(thresholds_df["safe_utilization"].max()),
            allowed_transfer_lanes=transfer_count,
            total_transfer_capacity_per_day=total_transfer_capacity,
            input_fingerprints=fingerprints,
        )
        return OperationalStateResponse(
            metadata=ServiceMetadata(
                service=self.service_name,
                correlation_id=context.correlation_id,
                warnings=[
                    "Snapshot reconstructs the bundled synthetic reference case from static tables; "
                    "it is not a live ADT/EHR state feed."
                ],
            ),
            state=snapshot,
        )
