from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import pandas as pd

from src.config.loader import load_healthcare_instance
from src.optimizer.model_builder import build_network_capacity_model
from src.scenarios.network_generator import generate_synthetic_network, write_synthetic_instance


DEFAULT_SIZES = [10, 25, 50, 100, 200]
RESULTS_PATH = Path("results/scalability_benchmark.csv")


def run_single_size(n_hospitals: int, time_limit_seconds: float) -> dict:
    costs_source = Path("data/base_instance/costs.csv")
    transitions_source = Path("data/transitions")

    print(f"--- N={n_hospitals} hospitals: generating synthetic network ---")
    tables = generate_synthetic_network(n_hospitals=n_hospitals)
    out_dir = Path(f"data/synthetic/n{n_hospitals}")
    write_synthetic_instance(tables, out_dir, costs_source, transitions_source)

    instance = load_healthcare_instance(data_dir=out_dir)
    n_lanes = len(instance.transfer_lanes.df)

    print(f"--- N={n_hospitals} hospitals: building and solving model (time limit {time_limit_seconds:.0f}s) ---")
    build_start = time.perf_counter()
    model, variables, validated_forecast_df, elective_response_df, arcs, transfer_cost_lookup = (
        build_network_capacity_model(instance=instance, model_name=f"scalability_n{n_hospitals}")
    )
    build_time = time.perf_counter() - build_start

    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit_seconds)

    solve_start = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - solve_start

    has_solution = model.SolCount > 0
    is_mip = bool(model.IsMIP)

    if has_solution and is_mip:
        mip_gap = float(model.MIPGap)
    elif has_solution:
        mip_gap = 0.0  # pure LP solved to proven optimality has no integrality gap
    else:
        mip_gap = None

    record = {
        "n_hospitals": n_hospitals,
        "n_transfer_lanes": n_lanes,
        "num_variables": int(model.NumVars),
        "num_constraints": int(model.NumConstrs),
        "is_mip": is_mip,
        "model_status": int(model.Status),
        "solution_count": int(model.SolCount),
        "objective_value": float(model.ObjVal) if has_solution else None,
        "mip_gap": mip_gap,
        "hit_time_limit": bool(model.Status == 9),  # Gurobi status 9 = TIME_LIMIT
        "build_time_seconds": build_time,
        "gurobi_solve_seconds": float(model.Runtime),
        "wall_clock_solve_seconds": solve_time,
    }

    print(record)
    return record


def run_benchmark(sizes: List[int], time_limit_seconds: float) -> pd.DataFrame:
    records = [run_single_size(n, time_limit_seconds) for n in sizes]
    new_df = pd.DataFrame(records)

    if RESULTS_PATH.exists():
        existing_df = pd.read_csv(RESULTS_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset="n_hospitals", keep="last")
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values("n_hospitals").reset_index(drop=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH} ({len(combined_df)} sizes total: {sorted(combined_df['n_hospitals'].tolist())})")
    return combined_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-hospital scalability benchmark.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Hospital network sizes to benchmark, e.g. --sizes 10 25 50",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="Per-solve Gurobi time limit in seconds (default: 120).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(sizes=args.sizes, time_limit_seconds=args.time_limit)


if __name__ == "__main__":
    main()