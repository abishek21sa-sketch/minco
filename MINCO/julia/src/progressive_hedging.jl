"""Progressive Hedging for scenario-decomposed MINCO first-stage decisions.

The implementation solves one scenario MILP/MIQP per iteration and penalizes
non-anticipativity deviations in surge and flex decisions.
"""
function _solve_ph_scenario(data::CapacityData, w::Int, xbar_surge, xbar_flex, u_surge, u_flex, rho::Float64; optimizer)
    H, T = length(data.hospitals), length(data.periods)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, surge[1:H,1:T], Bin)
    @variable(model, flex[1:H,1:T] >= 0, Int)
    @variable(model, defer[1:H,1:T] >= 0, Int)
    @variable(model, transfer[1:H,1:H,1:T] >= 0, Int)
    @variable(model, unsafe[1:H,1:T] >= 0)
    @variable(model, boarding[1:H,1:T] >= 0)
    @constraint(model, [h=1:H,t=1:T], flex[h,t] <= data.max_flex_blocks[h])
    @constraint(model, [h=1:H,t=1:T], defer[h,t] <= data.deferral_limit[h,t])
    @constraint(model, [i=1:H,j=1:H,t=1:T; i != j], transfer[i,j,t] <= data.transfer_capacity[i,j,t])
    @constraint(model, [h=1:H,t=1:T], transfer[h,h,t] == 0)
    @expression(model, capacity[h=1:H,t=1:T], data.base_beds[h] + data.surge_beds[h]*surge[h,t] + data.flex_beds_per_block[h]*flex[h,t])
    @expression(model, treated[h=1:H,t=1:T], data.demand[w,h,t] - defer[h,t] + sum(transfer[j,h,t] for j=1:H if j!=h) - sum(transfer[h,j,t] for j=1:H if j!=h))
    @constraint(model, [h=1:H,t=1:T], boarding[h,t] >= treated[h,t] - capacity[h,t])
    @constraint(model, [h=1:H,t=1:T], unsafe[h,t] >= treated[h,t] - data.safe_fraction[h]*capacity[h,t])
    @expression(model, base_cost,
        sum(data.surge_cost[h]*surge[h,t] + data.flex_cost[h]*flex[h,t] for h=1:H,t=1:T)
        + data.defer_cost*sum(defer)
        + data.transfer_cost*sum(transfer[i,j,t] for i=1:H,j=1:H,t=1:T if i!=j)
        + data.unsafe_cost*sum(unsafe)
        + data.boarding_cost*sum(boarding))
    @expression(model, penalty,
        sum(u_surge[h,t]*surge[h,t] + 0.5*rho*(surge[h,t]-xbar_surge[h,t])^2 for h=1:H,t=1:T)
        + sum(u_flex[h,t]*flex[h,t] + 0.5*rho*(flex[h,t]-xbar_flex[h,t])^2 for h=1:H,t=1:T))
    @objective(model, Min, base_cost + penalty)
    optimize!(model)
    has_values(model) || error("Progressive Hedging scenario $w failed: $(termination_status(model))")
    return value.(surge), value.(flex), objective_value(model)
end

function progressive_hedging(data::CapacityData; optimizer, rho::Float64=20.0, tolerance::Float64=1e-3, max_iterations::Int=60)
    _validate(data)
    H,T,W = length(data.hospitals), length(data.periods), length(data.probabilities)
    x_surge = zeros(W,H,T); x_flex = zeros(W,H,T)
    u_surge = zeros(W,H,T); u_flex = zeros(W,H,T)
    xbar_surge = zeros(H,T); xbar_flex = zeros(H,T)
    residual_history = Float64[]
    objective_history = Float64[]

    # Standard PH initialization: solve every scenario independently before
    # introducing the augmented non-anticipativity penalty.
    for w in 1:W
        x_surge[w,:,:], x_flex[w,:,:], _ = _solve_ph_scenario(
            data, w, xbar_surge, xbar_flex, u_surge[w,:,:], u_flex[w,:,:], 0.0; optimizer=optimizer)
    end
    xbar_surge .= 0.0
    xbar_flex .= 0.0
    for w in 1:W
        xbar_surge .+= data.probabilities[w] .* x_surge[w,:,:]
        xbar_flex .+= data.probabilities[w] .* x_flex[w,:,:]
    end

    for iteration in 1:max_iterations
        objectives = zeros(W)
        for w in 1:W
            x_surge[w,:,:], x_flex[w,:,:], objectives[w] = _solve_ph_scenario(
                data, w, xbar_surge, xbar_flex, u_surge[w,:,:], u_flex[w,:,:], rho; optimizer=optimizer)
        end
        xbar_surge .= 0.0
        xbar_flex .= 0.0
        for w in 1:W
            xbar_surge .+= data.probabilities[w] .* x_surge[w,:,:]
            xbar_flex .+= data.probabilities[w] .* x_flex[w,:,:]
        end
        residual = sqrt(sum(data.probabilities[w]*(sum((x_surge[w,:,:].-xbar_surge).^2)+sum((x_flex[w,:,:].-xbar_flex).^2)) for w=1:W))
        push!(residual_history, residual); push!(objective_history, sum(data.probabilities .* objectives))
        residual <= tolerance && return Dict("converged"=>true,"iterations"=>iteration,"residual_history"=>residual_history,"surge_consensus"=>round.(Int,xbar_surge),"flex_consensus"=>round.(Int,xbar_flex),"scenario_objective_history"=>objective_history)
        for w in 1:W
            u_surge[w,:,:] .+= rho .* (x_surge[w,:,:] .- xbar_surge)
            u_flex[w,:,:] .+= rho .* (x_flex[w,:,:] .- xbar_flex)
        end
    end
    return Dict("converged"=>false,"iterations"=>max_iterations,"residual_history"=>residual_history,"surge_consensus"=>round.(Int,xbar_surge),"flex_consensus"=>round.(Int,xbar_flex),"scenario_objective_history"=>objective_history)
end
