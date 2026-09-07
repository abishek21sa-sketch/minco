from src.benchmarks.stochastic_capacity import make_network_capacity_benchmark
from src.decision_math.stochastic_milp_oracle import (
    solve_small_stochastic_milp,
    validate_stochastic_solution,
)


def test_three_hospital_stochastic_benchmark_solves_and_is_independently_feasible():
    instance = make_network_capacity_benchmark(n_hospitals=3, n_periods=3, n_scenarios=6, seed=3)
    solution = solve_small_stochastic_milp(instance)
    checks = validate_stochastic_solution(instance, solution)
    assert solution.status == "OPTIMAL"
    assert checks["feasible"]
    assert checks["max_violation"] <= 1e-6
    assert solution.objective >= 0
