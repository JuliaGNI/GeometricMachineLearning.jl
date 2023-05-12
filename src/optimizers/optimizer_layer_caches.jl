abstract type AbstractLayerCache end

mutable struct AdamLayerCache{T, AT <: NamedTuple} <:AbstractLayerCache
    B₁::AT
    B₂::AT

    function AdamLayerCache(d::Lux.AbstractExplicitLayer)
        B₁, _ = Lux.setup(TrivialInitRNG(), d) .|> Lux.gpu
        B₂, _ = Lux.setup(TrivialInitRNG(), d) .|> Lux.gpu
        new{eltype(B₁), typeof(B₁)}(B₁, B₂)
    end
end

mutable struct MomentumLayerCache{T, AT <: NamedTuple} <:AbstractLayerCache
    B::AT

    function MomentumLayerCache(d::Lux.AbstractExplicitLayer)
        B, _ = Lux.setup(TrivialInitRNG(), d) .|> Lux.gpu
        new{eltype(B), typeof(B)}(B)
    end
end

struct StandardLayerCache <: AbstractLayerCache end
StandardLayerCache(::Lux.AbstractExplicitLayer) = StandardLayerCache()