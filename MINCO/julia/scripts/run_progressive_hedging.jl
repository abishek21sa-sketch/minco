import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "MINCOOptimization.jl"))
using .MINCOOptimization
using Gurobi
using JSON3
include(joinpath(@__DIR__, "reference_data.jl"))

data = tiny_capacity_data()
extensive = solve_extensive_form(data; optimizer=Gurobi.Optimizer, time_limit_seconds=30.0, mip_gap=1e-8)
ph = progressive_hedging(data; optimizer=Gurobi.Optimizer, rho=20.0, tolerance=1e-8, max_iterations=80)
agreement = (
    ph["converged"] == true &&
    ph["surge_consensus"] == extensive["surge"] &&
    ph["flex_consensus"] == extensive["flex_blocks"]
)
result = Dict(
    "acceptance" => agreement ? "PASSED" : "FAILED",
    "extensive_objective" => extensive["objective"],
    "extensive_surge" => extensive["surge"],
    "extensive_flex" => extensive["flex_blocks"],
    "ph_converged" => ph["converged"],
    "ph_iterations" => ph["iterations"],
    "ph_surge_consensus" => ph["surge_consensus"],
    "ph_flex_consensus" => ph["flex_consensus"],
    "residual_history" => ph["residual_history"],
)
println(JSON3.write(result))
agreement || exit(1)
