"""
Two-stage stochastic hospital network MILP with CVaR.

First-stage (scenario-independent):
  surge[h,t]       binary surge-pod activation
  flex[h,t]        integer flex-staff blocks

Second-stage recourse by scenario ω:
  defer[ω,h,t]     controllable elective demand deferred
  transfer[ω,i,j,t] patients routed between hospitals
  unsafe[ω,h,t]    census above safe staffed capacity
  boarding[ω,h,t]  census above physical/staffed capacity

The objective combines first-stage activation cost, expected recourse cost,
and CVaRα of scenario recourse loss.
"""
struct CapacityData
    hospitals::Vector{String}
    periods::Vector{Int}
    probabilities::Vector{Float64}
    demand::Array{Float64,3}              # ω,h,t
    base_beds::Vector{Float64}
    safe_fraction::Vector{Float64}
    surge_beds::Vector{Float64}
    surge_cost::Vector{Float64}
    flex_beds_per_block::Vector{Float64}
    flex_cost::Vector{Float64}
    max_flex_blocks::Vector{Int}
    deferral_limit::Array{Float64,2}       # h,t
    transfer_capacity::Array{Float64,3}    # i,j,t
    transfer_cost::Float64
    defer_cost::Float64
    unsafe_cost::Float64
    boarding_cost::Float64
    cvar_alpha::Float64
    cvar_weight::Float64
end

function _validate(data::CapacityData)
    H, T, W = length(data.hospitals), length(data.periods), length(data.probabilities)
    size(data.demand) == (W,H,T) || error("demand must have shape (scenario,hospital,period)")
    size(data.deferral_limit) == (H,T) || error("deferral_limit shape mismatch")
    size(data.transfer_capacity) == (H,H,T) || error("transfer_capacity shape mismatch")
    abs(sum(data.probabilities) - 1.0) <= 1e-9 || error("scenario probabilities must sum to one")
    all(data.probabilities .> 0) || error("scenario probabilities must be positive")
    all((0.0 .< data.safe_fraction) .& (data.safe_fraction .<= 1.0)) || error("invalid safe fractions")
    0.0 < data.cvar_alpha < 1.0 || error("cvar_alpha must be in (0,1)")
    return nothing
end

function solve_extensive_form(data::CapacityData; optimizer, time_limit_seconds::Float64=120.0, mip_gap::Float64=1e-4)
    _validate(data)
    H, T, W = length(data.hospitals), length(data.periods), length(data.probabilities)
    model = Model(optimizer)
    set_silent(model)
    try set_time_limit_sec(model, time_limit_seconds) catch end
    try set_optimizer_attribute(model, "MIPGap", mip_gap) catch end

    @variable(model, surge[1:H,1:T], Bin)
    @variable(model, flex[1:H,1:T] >= 0, Int)
    @variable(model, defer[1:W,1:H,1:T] >= 0, Int)
    @variable(model, transfer[1:W,1:H,1:H,1:T] >= 0, Int)
    @variable(model, unsafe[1:W,1:H,1:T] >= 0)
    @variable(model, boarding[1:W,1:H,1:T] >= 0)
    @variable(model, eta >= 0)
    @variable(model, xi[1:W] >= 0)

    @constraint(model, [h=1:H,t=1:T], flex[h,t] <= data.max_flex_blocks[h])
    @constraint(model, [w=1:W,h=1:H,t=1:T], defer[w,h,t] <= data.deferral_limit[h,t])
    @constraint(model, [w=1:W,i=1:H,j=1:H,t=1:T; i != j], transfer[w,i,j,t] <= data.transfer_capacity[i,j,t])
    @constraint(model, [w=1:W,h=1:H,t=1:T], transfer[w,h,h,t] == 0)

    @expression(model, physical_capacity[h=1:H,t=1:T],
        data.base_beds[h] + data.surge_beds[h]*surge[h,t] + data.flex_beds_per_block[h]*flex[h,t])
    @expression(model, treated[w=1:W,h=1:H,t=1:T],
        data.demand[w,h,t] - defer[w,h,t]
        + sum(transfer[w,j,h,t] for j=1:H if j != h)
        - sum(transfer[w,h,j,t] for j=1:H if j != h))

    @constraint(model, [w=1:W,h=1:H,t=1:T], boarding[w,h,t] >= treated[w,h,t] - physical_capacity[h,t])
    @constraint(model, [w=1:W,h=1:H,t=1:T], unsafe[w,h,t] >= treated[w,h,t] - data.safe_fraction[h]*physical_capacity[h,t])

    @expression(model, first_stage_cost,
        sum(data.surge_cost[h]*surge[h,t] + data.flex_cost[h]*flex[h,t] for h=1:H,t=1:T))
    @expression(model, scenario_loss[w=1:W],
        data.defer_cost*sum(defer[w,h,t] for h=1:H,t=1:T)
        + data.transfer_cost*sum(transfer[w,i,j,t] for i=1:H,j=1:H,t=1:T if i != j)
        + data.unsafe_cost*sum(unsafe[w,h,t] for h=1:H,t=1:T)
        + data.boarding_cost*sum(boarding[w,h,t] for h=1:H,t=1:T))
    @constraint(model, [w=1:W], xi[w] >= scenario_loss[w] - eta)
    @expression(model, expected_recourse, sum(data.probabilities[w]*scenario_loss[w] for w=1:W))
    @expression(model, cvar, eta + sum(data.probabilities[w]*xi[w] for w=1:W)/(1.0-data.cvar_alpha))
    @objective(model, Min, first_stage_cost + expected_recourse + data.cvar_weight*cvar)

    optimize!(model)
    term = termination_status(model)
    primal = primal_status(model)
    result = Dict{String,Any}(
        "termination_status" => string(term),
        "primal_status" => string(primal),
        "has_values" => has_values(model),
    )
    if has_values(model)
        result["objective"] = objective_value(model)
        result["surge"] = round.(Int, value.(surge))
        result["flex_blocks"] = round.(Int, value.(flex))
        result["scenario_loss"] = value.(scenario_loss)
        result["cvar"] = value(cvar)
        result["eta"] = value(eta)
        try result["relative_gap"] = MOI.get(model, MOI.RelativeGap()) catch; result["relative_gap"] = nothing end
        try result["solve_time_seconds"] = solve_time(model) catch; result["solve_time_seconds"] = nothing end
    end
    return result
end
