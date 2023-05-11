abstract type AbstractLayerCache end

mutable struct AdamLayerCache{T, AT <: NamedTuple} <:AbstractLayerCache
    B₁::AT
    B₂::AT

    function AdamLayerCache(d::Lux.AbstractExplicitLayer)
        B₁ = Lux.setup(TrivialInitRNG(), d)[1]
        B₂ = Lux.setup(TrivialInitRNG(), d)[1]
        new{eltype(B₁), typeof(B₁)}(B₁, B₂)
    end
end

mutable struct MomentumLayerCache{T, AT <: NamedTuple} <:AbstractLayerCache
    B::AT

    function MomentumLayerCache(d::Lux.AbstractExplicitLayer)
        B = Lux.setup(TrivialInitRNG(), d)[1]
        new{eltype(B), typeof(B)}(B)
    end
end

struct StandardLayerCache <: AbstractLayerCache end