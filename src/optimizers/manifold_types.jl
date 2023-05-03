"""
maybe consider dividing the output in the check functions by n!
TODO: Implement sampling procedures!!
"""

mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::AT
    function StiefelManifold(A::AbstractMatrix)
        @assert size(A)[1] ≥ size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
    #this draws a random element from U(StiefelManifold)
    function StiefelManifold(N::Int,n::Int)
        @assert N ≥ n
        A = randn(N,n)
        new{eltype(A), typeof(A)}(householderQ!(A))
    end
end

Base.size(A::StiefelManifold) = size(A.A)
Base.parent(A::StiefelManifold) = A.A 
Base.getindex(A::StiefelManifold, i::Int, j::Int) = A.A[i,j]


function check(A::StiefelManifold, tol=1e-10)
    @test norm(A'*A - I) < tol
    #print("Test passed.\n") 
end

mutable struct SymplecticStiefelManifold{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
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