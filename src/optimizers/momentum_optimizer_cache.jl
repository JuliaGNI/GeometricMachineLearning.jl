"""
Cache for the momentum optimizer. 
If the model is a simple layer, then this only saves the momentum term.
If it is a SymplecticStiefelLayer (or more general manifold layer), then it also saves the "previous step".
If it is a chain, then it saves two named tuples with momentum and prev_step information. 

TODO: Dispatch over optimizer! Define one type OptimizerCache!!
"""

function _init_cache(o::AbstractOptimizer, model::Lux.AbstractExplicitLayer, x::NamedTuple, dx::NamedTuple)
    o₂ = StandardOptimizer(o.η)
    update_layer!(o₂, nothing, model, x, dx)
    dx
end

mutable struct MomentumOptimizerLayerCache{MT <: NamedTuple,
                                           PT <: Union{Nothing, NamedTuple}} <:
               AbstractOptimizerCache
    momentum::MT
    prev_step::PT

    function MomentumOptimizerLayerCache(o::AbstractOptimizer,
                                         model::Lux.AbstractExplicitLayer,
                                         x::NamedTuple, dx::NamedTuple)
        momentum = _init_cache(o, model, x, dx)
        new{typeof(momentum), Nothing}(momentum, nothing)
    end

    function MomentumOptimizerLayerCache(o::AbstractOptimizer,
                                         model::SymplecticStiefelLayer, x::NamedTuple,
                                         dx::NamedTuple)
        prev_step = deepcopy(x)
        momentum = _init_cache(o, model, x, dx)
        new{typeof(momentum), typeof(prev_step)}(momentum, prev_step)
    end
end

#TODO: give this a different name than ``state'' - already used for an instance of AbstractOptimizerCache!!
mutable struct MomentumOptimizerCache <: AbstractOptimizerCache
    n_layer::Int
    state::NamedTuple

    function MomentumOptimizerCache(o::AbstractOptimizer, model::Lux.Chain, x::NamedTuple, dx::NamedTuple)
        state = NamedTuple()
        n_layer = length(model)
        for i in 1:n_layer
            layer_name = Symbol("layer_$i")
            state = merge(state,
                          NamedTuple{(layer_name,)}((MomentumOptimizerLayerCache(o,
                                                                                 model[i],
                                                                                 x[i],
                                                                                 dx[i]),)))
        end
        new(n_layer, state)
    end
end


#TODO: put this into another file and add an inital update!!!!
struct StandardOptimizerCache <: AbstractOptimizerCache
    a::Nothing
    function StandardOptimizerCache(::AbstractOptimizer, ::Lux.Chain, ::NamedTuple,
                                    ::NamedTuple)
        nothing
    end
end