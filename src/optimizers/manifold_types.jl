"""
maybe consider dividing the output in the check functions by n!
TODO: Implement sampling procedures!!
"""
abstract type Manifold <: AbstractMatrix end

mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold
    A::AT
    function StiefelManifold(A::AbstractMatrix)
        @assert size(A)[1] ≥ size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
end

Base.size(A::StiefelManifold) = size(A.A)
Base.parent(A::StiefelManifold) = A.A 
Base.getindex(A::StiefelManifold, i::Int, j::Int) = A.A[i,j]

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


function check(A::StiefelManifold, tol=1e-10)
    @test norm(A'*A - I) < tol
end

mutable struct SymplecticStiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold
    A::AT
    function SymplecticStiefelManifold(A::AbstractMatrix)
        @assert iseven(size(A)[1])
        @assert iseven(size(A)[2])
        @assert size(A)[1] ≥ size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
    #this should draw a random element from U(SymplecticStiefelManifold) -> doesn't work rn; implement using retractions!
    function SymplecticStiefelManifold(N::Int,n::Int)
        @assert N ≥ n
        A = randn(2*N,2*n)
        JN = SymplecticMatrix(N)
        Jn = SymplecticMatrix(n)
        new{eltype(A), typeof(A)}(A*inv(sqrt(Jn*A'*JN'*A)))
    end
end

Base.size(A::SymplecticStiefelManifold) = size(A.A)
Base.parent(A::SymplecticStiefelManifold) = A.A 
Base.getindex(A::SymplecticStiefelManifold, i::Int, j::Int) = A.A[i,j]


function check(A::SymplecticStiefelManifold, tol=1e-10)
    N = size(A)[1]÷2
    n = size(A)[2]÷2
    @test norm(A'*SymplecticMatrix(N)*A - SymplecticMatrix(n)) < tol
    #print("Test passed.\n") 
end