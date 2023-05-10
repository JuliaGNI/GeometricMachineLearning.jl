"""
maybe consider dividing the output in the check functions by n!
TODO: Implement sampling procedures!!
"""
mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold
    A::AT
    function StiefelManifold(A::AbstractMatrix)
        @assert size(A)[1] ≥ size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
end

#TODO: check the distribution this is coming from - related to the Haar measure ???
function Base.rand(rng::Random.AbstractRNG, ::Type{StiefelManifold{T}}, N::Int, n::Int) where T
    @assert N ≥ n
    A = randn(rng, T, N, n)
    StiefelManifold(qr(A).Q[1:N, 1:n])
end

function Base.rand(rng::TrivialInitRNG, ::Type{StiefelManifold{T}}, N::Int, n::Int) where T
    @assert N ≥ n 
    zeros(StiefelLieAlgHorMatrix{T}, N, n)
end

function rgrad(Y::StiefelManifold, e_grad::AbstractMatrix)
    e_grad - Y*(e_grad'*Y)
end

function check(A::StiefelManifold, tol=1e-10)
    @test norm(A'*A - I) < tol
end

function global_section(Y::StiefelManifold)
    N, n = size(Y)
    A = randn(eltype(Y), N-n, n)
    A = A - Y*Y'*A
    qr(A).Q[1:N, 1:N-n]
end