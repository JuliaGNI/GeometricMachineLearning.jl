
abstract type AbstractOptimizer end

#TODO specify type of state as AbstractOptimizerCache
#TODO: This only works for the StandardOptimizer, generalize at least to Momentum -> the state is the problem (state.state[i] for momentum & nothing for standrad)
function apply!(o::AbstractOptimizer, state, model::Lux.Chain, x::NamedTuple,
                dx::NamedTuple)
    for i in 1:length(model)
        #for i in eachindex(model, x, dx)
        update_layer!(o, state, model[i], x[i], dx[i])
    end
end

function apply!(o::AbstractOptimizer, state::AbstractOptimizerCache, model::Lux.Chain,
                x::NamedTuple, dx::NamedTuple)
    for i in 1:length(model)
        #layer_name = Symbol("layer_$i")
        update_layer!(o, state.state[i], model[i], x[i], dx[i])
    end
end