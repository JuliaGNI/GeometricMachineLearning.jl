abstract type AbstractCache end

mutable struct AdamCache{T, AT <: AbstractMatrix} <: AbstractCache
    B₁::AT
    B₂::AT 
    function AdamCache(B₁::AbstractMatrix, B₂::AbstractMatrix)
        new{eltype(B₁), typeof(B₁)}(B₁, B₂)
    end
end

mutable struct MomentumCache{T, AT <: NamedTuple} <:AbstractCache
    B::AT
    function MomentumCache(B::AbstractMatrix)
        new{eltype(B), typeof(B)}(B)
    end
end

struct StandardCache <: AbstractCache end
StandardLayerCache(::AbstractMatrix) = StandardCache()

function AdamCache(dx₁::NamedTuple, dx₂::NamedTuple)
    apply_toNT(dx₁, dx₂, AdamCache)
end

function MomentumCache(dx::NamedTuple)
    apply_toNT(dx, MomentumCache)
end

function StandardCache(dx::NamedTuple)
    apply_toNT(dx, StandardLayerCache)
end

function AdamCache(d::Lux.AbstractExplicitLayer)
    B₁ = Lux.setup(TrivialInitRNG(), d) .|> gpu
    B₂ = Lux.setup(TrivialInitRNG(), d) .|> gpu
    AdamCache(B₁, B₂)
end

function MomentumCache(d::Lux.AbstractExplicitLayer)
    B = Lux.setup(TrivialInitRNG(), d) .|> gpu
    MomentumCache(B)
end

#TODO: make this more efficient! you don't need to call the entire setup to initialize an empty NamedTuple!!!
function StandardCache(d::Lux.AbstractExplicitLayer)
    B = Lux.setup(TrivialInitRNG(), d) .|> gpu
    StandardLayerCache(B)
end