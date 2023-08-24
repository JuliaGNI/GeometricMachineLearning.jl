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

function Base.rand(backend::KernelAbstractions.Backend, rng::Random.AbstractRNG, ::Type{GrassmannManifold{T}}, N::Integer, n::Integer) where T
    @assert N ≥ n 
    A = KernelAbstractions.allocate(backend, T, N, n)
    Random.randn!(rng, A)
    GrassmannManifold(A)
end


function rgrad(Y::GrassmannManifold, e_grad::AbstractMatrix)
    e_grad - Y*inv(Y'*Y)*(Y'*e_grad)
end

function metric(Y::GrassmannManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    LinearAlgebra.tr(Δ₁'*(I - Y*inv(Y'*Y)*Y')*Δ₂)
end

function global_section(Y::GrassmannManifold)
    N, n = size(Y)
    A = randn(eltype(Y), N, N-n)
    #A - Y*inv(Y'*Y)*Y'*A
    qr!(hcat(Y, A)).Q
end
