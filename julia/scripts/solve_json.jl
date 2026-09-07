import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "MINCOOptimization.jl"))
using .MINCOOptimization
using Gurobi
using JSON3

length(ARGS) == 2 || error("Usage: julia solve_json.jl input.json output.json")
input_path, output_path = ARGS[1], ARGS[2]
obj = JSON3.read(read(input_path, String))
H = length(obj.hospitals); T = length(obj.periods); W = length(obj.probabilities)

demand = zeros(Float64, W,H,T)
for w in 1:W, h in 1:H, t in 1:T
    demand[w,h,t] = Float64(obj.demand[w][h][t])
end
deferral = zeros(Float64,H,T)
for h in 1:H, t in 1:T
    deferral[h,t] = Float64(obj.deferral_limit[h][t])
end
transfer = zeros(Float64,H,H,T)
for i in 1:H, j in 1:H, t in 1:T
    transfer[i,j,t] = Float64(obj.transfer_capacity[i][j][t])
end

data = CapacityData(
    String.(obj.hospitals), Int.(obj.periods), Float64.(obj.probabilities), demand,
    Float64.(obj.base_beds), Float64.(obj.safe_fraction), Float64.(obj.surge_beds),
    Float64.(obj.surge_cost), Float64.(obj.flex_beds_per_block), Float64.(obj.flex_cost),
    Int.(obj.max_flex_blocks), deferral, transfer,
    Float64(obj.transfer_cost), Float64(obj.defer_cost), Float64(obj.unsafe_cost),
    Float64(obj.boarding_cost), Float64(obj.cvar_alpha), Float64(obj.cvar_weight),
)
result = solve_extensive_form(data; optimizer=Gurobi.Optimizer)
write(output_path, JSON3.write(result))
println(JSON3.write(Dict("status"=>result["termination_status"], "objective"=>get(result,"objective",nothing))))
