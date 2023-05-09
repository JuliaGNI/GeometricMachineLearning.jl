abstract type AbstractLayerCache end

struct TrivialInitRNG{T<:AbstractFloat} <: Random.AbstractRNG
    state::T
    function TrivialInitRNG(T::Type{FT}=Float32) where FT<:Union{AbstractFloat, Integer}
        new{T}(zero(T))
    end
end
function Random.seed!(r::TrivialInitRNG) 
end
function Random.rng_native_52(::TrivialInitRNG{T}) where T <: Union{AbstractFloat, Integer}
    T
end

function Base.rand(::TrivialInitRNG, T::Type{X}, d::Integer, dims::Integer...) where X
    zeros(T, d, dims...)
end

#Lux is always working with single precision!
function Lux.glorot_uniform(rng::TrivialInitRNG, dims::Integer...; gain = 1)
    rand(rng, Float32, dims...)
end

mutable struct AdamLayerCache{T, AT <: NamedTuple} <:AbstractLayerCache
    B₁::AT
    B₂::AT

    function AdamLayerCache(d::Lux.AbstractExplicitLayer)
        B₁ = Lux.setup(TrivialInitRNG(), d)
        B₂ = Lux.setup(TrivialInitRNG(), d)
        new{eltype(B₁, B₁)}(B₁, B₂)
    end
end

mutable struct MomentumLayerCache{T, AT <: NamedTuple} <:AbstractLayerCache
    B::AT

    function MomentumLayerCache(d::Lux.AbstractExplicitLayer)
        B = Lux.setup(TrivialRNG(), d)
        new{eltype(B)}(B)
    end
end

struct StandardLayerCache <: AbstractLayerCache
    B::Nothing
    function StandardLayerCache(d::Lux.AbstractExplicitLayer)
        new(nothing)
    end
end