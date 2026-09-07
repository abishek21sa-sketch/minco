"""Patient-flow conservation and staffed-capacity mathematics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowBalance:
    opening_census: float
    arrivals: float
    transfers_in: float
    discharges: float
    transfers_out: float
    closing_census: float
    residual: float


def close_census(
    *,
    opening_census: float,
    arrivals: float,
    transfers_in: float,
    discharges: float,
    transfers_out: float,
) -> float:
    values = [opening_census, arrivals, transfers_in, discharges, transfers_out]
    if any(v < 0 for v in values):
        raise ValueError("Flow quantities must be nonnegative")
    closing = opening_census + arrivals + transfers_in - discharges - transfers_out
    if closing < -1e-9:
        raise ValueError("Patient flow would produce a negative census")
    return float(max(0.0, closing))


def verify_flow_balance(
    *,
    opening_census: float,
    arrivals: float,
    transfers_in: float,
    discharges: float,
    transfers_out: float,
    observed_closing_census: float,
) -> FlowBalance:
    expected = close_census(
        opening_census=opening_census,
        arrivals=arrivals,
        transfers_in=transfers_in,
        discharges=discharges,
        transfers_out=transfers_out,
    )
    residual = float(observed_closing_census - expected)
    return FlowBalance(
        opening_census=float(opening_census),
        arrivals=float(arrivals),
        transfers_in=float(transfers_in),
        discharges=float(discharges),
        transfers_out=float(transfers_out),
        closing_census=float(observed_closing_census),
        residual=residual,
    )


def maximum_staffed_beds(
    *,
    available_nurse_hours: float,
    nurse_hours_per_bed: float,
    physical_beds: float,
) -> float:
    if available_nurse_hours < 0 or nurse_hours_per_bed <= 0 or physical_beds < 0:
        raise ValueError("Invalid staffing/capacity inputs")
    staff_limited = available_nurse_hours / nurse_hours_per_bed
    return float(min(physical_beds, staff_limited))


def staffing_adequacy_ratio(
    *,
    available_nurse_hours: float,
    required_nurse_hours: float,
) -> float:
    if available_nurse_hours < 0 or required_nurse_hours < 0:
        raise ValueError("Nurse-hour values must be nonnegative")
    if required_nurse_hours == 0:
        return 1.0
    return float(available_nurse_hours / required_nurse_hours)
