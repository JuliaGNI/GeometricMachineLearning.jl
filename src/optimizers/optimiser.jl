
#######################################################################################
#MethodOptimiser

abstract type AbstractMethodOptimiser end

abstract type AbstractOptimizerCache end

const OptimizerCache = Union{AbstractOptimizerCache,Missing}

mutable struct GradientOptimizer{T} <: AbstractMethodOptimiser
    η::T
    GradientOptimizer(η = 1e-2) = new{typeof(η)}(η)
end

#######################################################################################
#Optimiser

struct Optimiser{MT<:AbstractMethodOptimiser, CT<:OptimizerCache}
    method::MT
    cache::CT

    function Optimiser(m::AbstractMethodOptimiser)
        cache = init_cache(m)
        new{typeof(m),typeof(cache)}(m,cache)
    end
end

#######################################################################################
#init_cache

init_cache(m::GradientOptimizer) = missing

#######################################################################################
#apply
#=
function apply!(o::Optimizer, model::Lux.Chain, x::NamedTuple, dx::NamedTuple)
    for i in 1:length(model)
        #for i in eachindex(model, x, dx)
        update_layer!(o, Nothing, model[i], x[i], dx[i])
    end
end

function apply!(o::Optimizer_w_Cache, model::Lux.Chain, x::NamedTuple, dx::NamedTuple)
    for i in 1:length(model)
        #layer_name = Symbol("layer_$i")
        update_layer!(o, o.cache.state[i], model[i], x[i], dx[i])
    end
end
=#