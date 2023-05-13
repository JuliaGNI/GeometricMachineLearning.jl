
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


