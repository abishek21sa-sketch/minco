import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "MINCOOptimization.jl"))
using .MINCOOptimization
using Gurobi
using JSON3
include(joinpath(@__DIR__, "reference_data.jl"))

data = tiny_capacity_data()
result = solve_extensive_form(data; optimizer=Gurobi.Optimizer, time_limit_seconds=30.0, mip_gap=1e-8)
result["acceptance"] = (
    result["termination_status"] == "OPTIMAL" &&
    result["has_values"] == true &&
    abs(result["objective"] - 12.86) <= 1e-5
) ? "PASSED" : "FAILED"
println(JSON3.write(result))
result["acceptance"] == "PASSED" || exit(1)
