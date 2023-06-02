"""
This implements the horizontal component of the Lie algebra (in this case just the skew-symmetric matrices).
The projection is: 
S -> SE where 
|I|
|0| = E.

An element of GrassmannLieAlgMatrix takes the form: 
| -0 -B'|
| B  0 | where B is arbitrary.

This also implements the projection: 
| 0 -B'|    | 0 -B'|
| B  0 | -> | B  0 |.
"""

mutable struct GrassmanLieAlgHorMatrix{T, ST <: AbstractMatrix{T}} <: AbstractMatrix{T}
    B::ST
    N::Int
    n::Int 

    #maybe modify this - you don't need N & n as inputs!
    function GrassmanLieAlgHorMatrix(B::AbstractMatrix{T}, N::Int, n::Int) where {T}
        @assert n == size(B,2) 
        @assert N == size(B,1) + n

        new{T, typeof(B)}(B, N, n)
    end 

    function GrassmanLieAlgHorMatrix(A::SkewSymMatrix{T}, n::Int) where {T}
        N = A.n 
        @assert N ≥ n 

        B = A[(n+1):N,1:n]
        new{eltype(A),typeof(A), typeof(B)}(A_small, B, N, n)
    end
end 

Base.parent(A::GrassmanLieAlgHorMatrix) = (B)
Base.size(A::GrassmanLieAlgHorMatrix) = (A.N, A.N)

function Base.getindex(A::GrassmanLieAlgHorMatrix{T}, i::Integer, j::Integer) where {T}
    if i ≤ A.n
        if j ≤ A.n 
            return T(0.)
        end
        return -A.B[j - A.n, i]
    end
    if j ≤ A.n 
        return A.B[i - A.n, j]
    end
    return T(0.)
end

function Base.:+(A::GrassmanLieAlgHorMatrix, B::GrassmanLieAlgHorMatrix)
    @assert A.N == B.N 
    @assert A.n == B.n 
    GrassmanLieAlgHorMatrix(A.B + B.B, 
                            A.N,
                            A.n)
end

function Base.:-(A::GrassmanLieAlgHorMatrix, B::GrassmanLieAlgHorMatrix)
    @assert A.N == B.N 
    @assert A.n == B.n 
    GrassmanLieAlgHorMatrix(A.B - B.B, 
                            A.N,
                            A.n)
end

function add!(C::GrassmanLieAlgHorMatrix, A::GrassmanLieAlgHorMatrix, B::GrassmanLieAlgHorMatrix)
    @assert A.N == B.N == C.N
    @assert A.n == B.n == C.n 
    add!(C.B, A.B, B.B)  
end

function Base.:-(A::GrassmanLieAlgHorMatrix)
    GrassmanLieAlgHorMatrix( -A.B, A.N, A.n)
end

function Base.:*(A::GrassmanLieAlgHorMatrix, α::Real)
    GrassmanLieAlgHorMatrix( α*A.B, A.N, A.n)
end

Base.:*(α::Real, A::GrassmanLieAlgHorMatrix) = A*α

function Base.zeros(::Type{GrassmanLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    GrassmanLieAlgHorMatrix(
        zeros(T, N-n, n),
        N, 
        n
    )
end
    
function Base.zeros(::Type{GrassmanLieAlgHorMatrix}, N::Integer, n::Integer)
    GrassmanLieAlgHorMatrix(
        zeros(N-n, n),
        N, 
        n
    )
end

function Base.rand(rng::Random.AbstractRNG, ::Type{GrassmanLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    GrassmanLieAlgHorMatrix(rand(rng, T, N-n, n), N, n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{GrassmanLieAlgHorMatrix}, N::Integer, n::Integer)
    GrassmanLieAlgHorMatrix(rand(rng, N-n, n), N, n)
end

function Base.rand(::Type{GrassmanLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    rand(Random.default_rng(), GrassmanLieAlgHorMatrix{T}, N, n)
end

function Base.rand(::Type{GrassmanLieAlgHorMatrix}, N::Integer, n::Integer)
    rand(Random.default_rng(), GrassmanLieAlgHorMatrix, N, n)
end

function scalar_add(A::GrassmanLieAlgHorMatrix, δ::Real)
    GrassmanLieAlgHorMatrix(A.B .+ δ, A.N, A.n)
end

#define these functions more generally! (maybe make a fallback script!!)
function ⊙²(A::GrassmanLieAlgHorMatrix)
    GrassmanLieAlgHorMatrix(A.B.^2, A.N, A.n)
end
function √ᵉˡᵉ(A::GrassmanLieAlgHorMatrix)
    GrassmanLieAlgHorMatrix(sqrt.(A.B), A.N, A.n)
end
function /ᵉˡᵉ(A::GrassmanLieAlgHorMatrix, B::GrassmanLieAlgHorMatrix)
    GrassmanLieAlgHorMatrix(A.B./B.B, A.N, A.n)
end 

function LinearAlgebra.mul!(C::GrassmanLieAlgHorMatrix, A::GrassmanLieAlgHorMatrix, α::Real)
    mul!(C.B, A.B, α)
end
LinearAlgebra.mul!(C::GrassmanLieAlgHorMatrix, α::Real, A::GrassmanLieAlgHorMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::GrassmanLieAlgHorMatrix, α::Real) = mul!(C, C, α)