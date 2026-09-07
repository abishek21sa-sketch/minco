from __future__ import annotations

from pathlib import Path
from typing import Dict

from dashboard.live_sandbox import load_base_instance, run_live_whatif


def get_current_state(
    data_dir: Path = Path("data"),
    icu_bed_delta: int = 0,
    transfer_capacity_multiplier: float = 1.0,
    demand_surge_multiplier: float = 1.0,
    n_replications: int = 50,
) -> Dict[str, float]:
    """
    Run a live solve against the current network state and return the
    resulting KPI snapshot. This reuses the exact same live solver as the
    Scenario Lab / Copilot / Event Simulator -- the Command Center's "current
    state" reading is a genuine live solve, not a cached report file.
    """
    instance = load_base_instance(data_dir)
    return run_live_whatif(
        instance=instance,
        icu_bed_delta=icu_bed_delta,
        transfer_capacity_multiplier=transfer_capacity_multiplier,
        demand_surge_multiplier=demand_surge_multiplier,
        n_replications=n_replications,
    )