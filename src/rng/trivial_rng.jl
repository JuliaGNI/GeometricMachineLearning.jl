"""
A trivial RNG that is used to initialize the optimizers. 
"""

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