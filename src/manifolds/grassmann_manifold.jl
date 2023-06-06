"""
maybe consider dividing the output in the check functions by n!
TODO: Implement sampling procedures!!
"""
mutable struct GrassmannManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
    function GrassmannManifold(A::AbstractMatrix)
        @assert size(A)[1] ≥ size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
end

#TODO: check the distribution this is coming from - related to the Haar measure ???
function Base.rand(rng::Random.AbstractRNG, ::Type{GrassmannManifold{T}}, N::Integer, n::Integer) where T
    @assert N ≥ n
    GrassmannManifold(randn(rng, T, N, n))
end

function Base.rand(rng::Random.AbstractRNG, ::Type{GrassmannManifold}, N::Integer, n::Integer)
    @assert N ≥ n 
    GrassmannManifold(randn(rng, N, n))
end

function Base.rand(::Type{GrassmannManifold{T}}, N::Integer, n::Integer) where T
    @assert N ≥ n
    GrassmannManifold(randn(T, N, n))
end

function Base.rand(::Type{GrassmannManifold}, N::Integer, n::Integer)
    @assert N ≥ n 
    GrassmannManifold(randn(N, n))
end

#function Base.rand(::TrivialInitRNG, ::Type{GrassmannManifold{T}}, N::Int, n::Int) where T
#@assert N ≥ n 
#    zeros(StiefelLieAlgHorMatrix{T}, N, n)
#end

function Base.rand(::TrivialInitRNG{T}, ::Type{GrassmannManifold}, N::Int, n::Int) where {T<:AbstractFloat}
    @assert N ≥ n 
    zeros(GrassmannLieAlgHorMatrix{T}, N, n)
end

function rgrad(Y::GrassmannManifold, e_grad::AbstractMatrix)
    e_grad - Y*Y'*e_grad'
end

function metric(Y::GrassmannManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    LinearAlgebra.tr(Δ₁'*(I - Y*Y')*Δ₂)
end

function global_section(Y::GrassmannManifold)
    N, n = size(Y)
    A = randn(eltype(Y), N, N-n)
    A - Y*inv(Y'*Y)*Y'*A
end

function global_section(::AbstractVecOrMat)
    nothing
end