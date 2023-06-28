"""
This implements the horizontal component of the Lie algebra (in this case just the skew-symmetric matrices).
The projection is: 
S -> SE where 
|I|
|0| = E.

An element of StiefelLieAlgMatrix takes the form: 
| A -B'|
| B  0 | where A is skew-symmetric.

This also implements the projection: 
| A -B'|    | A -B'|
| B  D | -> | B  0 |.
"""

mutable struct StiefelLieAlgHorMatrix{T, AT <: SkewSymMatrix{T}, ST <: AbstractMatrix{T}} <: AbstractMatrix{T}
    A::AT
    B::ST
    N::Int
    n::Int 

    #maybe modify this - you don't need N & n as inputs!
    function StiefelLieAlgHorMatrix(A::SkewSymMatrix{T}, B::AbstractMatrix{T}, N::Int, n::Int) where {T}
        @assert n == A.n == size(B,2) 
        @assert N == size(B,1) + n

        new{T, typeof(A), typeof(B)}(A, B, N, n)
    end 

    function StiefelLieAlgHorMatrix(A::AbstractMatrix, n::Int)
        N = size(A, 1)
        @assert N ≥ n 

        A_small = 2*SkewSymMatrix(A[1:n,1:n])
        B = A[(n+1):N,1:n]
        new{eltype(A),typeof(A), typeof(B)}(A_small, B, N, n)
    end
end 

Base.parent(A::StiefelLieAlgHorMatrix) = (A, B)
Base.size(A::StiefelLieAlgHorMatrix) = (A.N, A.N)

function Base.getindex(A::StiefelLieAlgHorMatrix{T}, i, j) where {T}
    if i ≤ A.n
        if j ≤ A.n 
            return A.A[i, j]
        end
        return -A.B[j - A.n, i]
    end
    if j ≤ A.n 
        return A.B[i - A.n, j]
    end
    return T(0.)
end

function Base.:+(A::StiefelLieAlgHorMatrix, B::StiefelLieAlgHorMatrix)
    @assert A.N == B.N 
    @assert A.n == B.n 
    StiefelLieAlgHorMatrix( A.A + B.A, 
                            A.B + B.B, 
                            A.N,
                            A.n)
end

function Base.:-(A::StiefelLieAlgHorMatrix, B::StiefelLieAlgHorMatrix)
    @assert A.N == B.N 
    @assert A.n == B.n 
    StiefelLieAlgHorMatrix( A.A - B.A, 
                            A.B - B.B, 
                            A.N,
                            A.n)
end

function add!(C::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix, B::StiefelLieAlgHorMatrix)
    @assert A.N == B.N == C.N
    @assert A.n == B.n == C.n 
    add!(C.A, A.A, B.A) 
    add!(C.B, A.B, B.B)  
end


function Base.:-(A::StiefelLieAlgHorMatrix)
    StiefelLieAlgHorMatrix(-A.A, -A.B, A.N, A.n)
end

function Base.:*(A::StiefelLieAlgHorMatrix, α::Real)
    StiefelLieAlgHorMatrix( α*A.A, α*A.B, A.N, A.n)
end

Base.:*(α::Real, A::StiefelLieAlgHorMatrix) = A*α

function Base.zeros(::Type{StiefelLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    StiefelLieAlgHorMatrix(
        zeros(SkewSymMatrix{T}, n),
        zeros(T, N-n, n),
        N, 
        n
    )
end
    
function Base.zeros(::Type{StiefelLieAlgHorMatrix}, N::Integer, n::Integer)
    StiefelLieAlgHorMatrix(
        zeros(SkewSymMatrix, n),
        zeros(N-n, n),
        N, 
        n
    )
end

function Base.rand(rng::Random.AbstractRNG, ::Type{StiefelLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    StiefelLieAlgHorMatrix(rand(rng, SkewSymMatrix{T}, n), rand(rng, T, N-n, n), N, n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{StiefelLieAlgHorMatrix}, N::Integer, n::Integer)
    StiefelLieAlgHorMatrix(rand(rng, SkewSymMatrix, n), rand(rng, N-n, n), N, n)
end

function Base.rand(::Type{StiefelLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    rand(Random.default_rng(), StiefelLieAlgHorMatrix{T}, N, n)
end

function Base.rand(::Type{StiefelLieAlgHorMatrix}, N::Integer, n::Integer)
    rand(Random.default_rng(), StiefelLieAlgHorMatrix, N, n)
end

function scalar_add(A::StiefelLieAlgHorMatrix, δ::Real)
    StiefelLieAlgHorMatrix(scalar_add(A.A, δ), A.B .+ δ, A.N, A.n)
end

#define these functions more generally! (maybe make a fallback script!!)
function ⊙²(A::StiefelLieAlgHorMatrix)
    StiefelLieAlgHorMatrix(⊙²(A.A), A.B.^2, A.N, A.n)
end
function racᵉˡᵉ(A::StiefelLieAlgHorMatrix)
    StiefelLieAlgHorMatrix(racᵉˡᵉ(A.A), sqrt.(A.B), A.N, A.n)
end
function /ᵉˡᵉ(A::StiefelLieAlgHorMatrix, B::StiefelLieAlgHorMatrix)
    StiefelLieAlgHorMatrix(/ᵉˡᵉ(A.A, B.A), A.B./B.B, A.N, A.n)
end 

function LinearAlgebra.mul!(C::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix, α::Real)
    mul!(C.A, A.A, α)
    mul!(C.B, A.B, α)
end
LinearAlgebra.mul!(C::StiefelLieAlgHorMatrix, α::Real, A::StiefelLieAlgHorMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::StiefelLieAlgHorMatrix, α::Real) = mul!(C, C, α)

