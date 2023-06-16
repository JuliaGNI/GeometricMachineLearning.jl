#This files contains Cache's structure

abstract type AbstractCache end

#############################################################################
# All the definitions of the caches

mutable struct AdamCache{T, AT <: AbstractMatrix} <: AbstractCache
    B₁::AT
    B₂::AT 
    function AdamCache(B₁::AbstractMatrix, B₂::AbstractMatrix)
        new{eltype(B₁), typeof(B₁)}(B₁, B₂)
    end
end

mutable struct MomentumCache{T, AT <: AbstractMatrix} <:AbstractCache
    B::AT
    function MomentumCache(B::AbstractMatrix)
        new{eltype(B), typeof(B)}(B)
    end
end

struct GradientCache <: AbstractCache end
GradientCache(::AbstractMatrix) = GradientCache()

#############################################################################
# All the setup_cache functions 

setup_adam_cache(B₁::NamedTuple, B₂::NamedTuple) = apply_toNT(B₁, B₂, setup_adam_cache)
setup_momentum_cache(dx::NamedTuple) = apply_toNT(dx, setup_momentum_cache)
setup_gradient_cache(dx::NamedTuple) = apply_toNT(dx, setup_gradient_cache)

setup_adam_cache(B₁::AbstractMatrix, B₂::AbstractMatrix) = AdamCache(B₁, B₂)
setup_momentum_cache(B::AbstractMatrix) = MomentumCache(B)
setup_gradient_cache(B::AbstractMatrix) = GradientCache(B)

function setup_adam_cache(d::Lux.AbstractExplicitLayer)
    B₁, _ = Lux.setup(TrivialInitRNG(), d) #.|> gpu
    B₂, _ = Lux.setup(TrivialInitRNG(), d) #.|> gpu
    setup_adam_cache(B₁, B₂)
end

function setup_momentum_cache(d::Lux.AbstractExplicitLayer)
    B, _ = Lux.setup(TrivialInitRNG(), d) #.|> gpu
    setup_momentum_cache(B)
end

function setup_gradient_cache(d::Lux.AbstractExplicitLayer)
    B, _ = Lux.setup(TrivialInitRNG(), d) #.|> gpu
    setup_gradient_cache(B)
end
