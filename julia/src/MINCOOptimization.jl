module MINCOOptimization

using JuMP
import MathOptInterface as MOI

include("extensive_form.jl")
include("progressive_hedging.jl")
include("rolling_horizon.jl")

export CapacityData,
       solve_extensive_form,
       progressive_hedging,
       rolling_horizon_solve

end
