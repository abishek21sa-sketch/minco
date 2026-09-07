"""Rolling-horizon reoptimization wrapper.

At each decision epoch the data callback supplies the latest stochastic demand
scenarios. Only the first period's first-stage action is committed, after which
the horizon advances and the problem is rebuilt with refreshed information.
"""
function rolling_horizon_solve(data_callback, epochs; optimizer, horizon_periods::Int=6)
    committed = Vector{Dict{String,Any}}()
    for epoch in epochs
        data = data_callback(epoch, horizon_periods)
        result = solve_extensive_form(data; optimizer=optimizer)
        result["has_values"] || error("Rolling-horizon solve failed at epoch $epoch")
        push!(committed, Dict(
            "epoch"=>epoch,
            "surge_first_period"=>result["surge"][:,1],
            "flex_first_period"=>result["flex_blocks"][:,1],
            "objective"=>result["objective"],
            "relative_gap"=>get(result,"relative_gap",nothing),
        ))
    end
    return committed
end
