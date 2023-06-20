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
setup_gradient_cache(B::AbstractMatrix) = StandardCache(B)

function setup_adam_cache(dev::Device, d::Lux.AbstractExplicitLayer)
    B₁, _ = Lux.setup(dev, TrivialInitRNG(), d)
    B₂, _ = Lux.setup(dev, TrivialInitRNG(), d)
    setup_adam_cache(B₁, B₂)
end

function setup_momentum_cache(dev::Device, d::Lux.AbstractExplicitLayer)
    B, _ = Lux.setup(dev, TrivialInitRNG(), d) 
    setup_momentum_cache(B)
end

#TODO: make this more efficient! you don't need to call the entire setup to initialize an empty NamedTuple!!!
function setup_gradient_cache(dev::Device, d::Lux.AbstractExplicitLayer)
    B, _ = Lux.setup(dev, TrivialInitRNG(), d) 
    setup_gradient_cache(B)
end
