
abstract type AbstractOptimizer end


function apply!(o::AbstractOptimizer, cache::Nothing, model::Lux.Chain, x::NamedTuple,
                dx::NamedTuple)
    for i in 1:length(model)
        #for i in eachindex(model, x, dx)
        update_layer!(o, cache, model[i], x[i], dx[i])
    end
end

function apply!(o::AbstractOptimizer, cache::AbstractOptimizerCache, model::Lux.Chain,
                x::NamedTuple, dx::NamedTuple)
    for i in 1:length(model)
        #layer_name = Symbol("layer_$i")
        update_layer!(o, cache.state[i], model[i], x[i], dx[i])
    end
end