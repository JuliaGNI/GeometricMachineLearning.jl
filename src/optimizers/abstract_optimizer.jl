
abstract type AbstractOptimizer end

abstract type AbstractOptimizer_w_Cache <: AbstractOptimizer end

function setup_Optimiser!(o::AbstractOptimizer, model::Lux.Chain, x::NamedTuple, ∇Loss::Function)
    error("setup_Optimiser not implemented for layer type ", typeof(o))
end

function apply!(o::AbstractOptimizer, model::Lux.Chain, x::NamedTuple, dx::NamedTuple)
    for i in 1:length(model)
        #for i in eachindex(model, x, dx)
        update_layer!(o, Nothing, model[i], x[i], dx[i])
    end
end

function apply!(o::AbstractOptimizer_w_Cache, model::Lux.Chain, x::NamedTuple, dx::NamedTuple)
    for i in 1:length(model)
        #layer_name = Symbol("layer_$i")
        update_layer!(o, o.cache.state[i], model[i], x[i], dx[i])
    end
end