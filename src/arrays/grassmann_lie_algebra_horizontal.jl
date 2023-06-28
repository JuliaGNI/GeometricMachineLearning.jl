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

mutable struct GrassmannLieAlgHorMatrix{T, ST <: AbstractMatrix{T}} <: AbstractMatrix{T}
    B::ST
    N::Int
    n::Int 

    #maybe modify this - you don't need N & n as inputs!
    function GrassmannLieAlgHorMatrix(B::AbstractMatrix{T}, N::Int, n::Int) where {T}
        @assert n == size(B,2) 
        @assert N == size(B,1) + n

        new{T, typeof(B)}(B, N, n)
    end 

    function GrassmannLieAlgHorMatrix(A::AbstractMatrix{T}, n::Int) where {T}
        N = size(A, 1)
        @assert N ≥ n 

        B = A[(n+1):N,1:n]
        new{eltype(A), typeof(B)}(B, N, n)
    end
end 

Base.parent(A::GrassmannLieAlgHorMatrix) = (B)
Base.size(A::GrassmannLieAlgHorMatrix) = (A.N, A.N)

function Base.getindex(A::GrassmannLieAlgHorMatrix{T}, i::Integer, j::Integer) where {T}
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

function Base.:+(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N 
    @assert A.n == B.n 
    GrassmannLieAlgHorMatrix(A.B + B.B, 
                            A.N,
                            A.n)
end

function Base.:-(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N 
    @assert A.n == B.n 
    GrassmannLieAlgHorMatrix(A.B - B.B, 
                            A.N,
                            A.n)
end

function add!(C::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N == C.N
    @assert A.n == B.n == C.n 
    add!(C.B, A.B, B.B)  
end

function Base.:-(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix( -A.B, A.N, A.n)
end

function Base.:*(A::GrassmannLieAlgHorMatrix, α::Real)
    GrassmannLieAlgHorMatrix( α*A.B, A.N, A.n)
end

Base.:*(α::Real, A::GrassmannLieAlgHorMatrix) = A*α

function Base.zeros(::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    GrassmannLieAlgHorMatrix(
        zeros(T, N-n, n),
        N, 
        n
    )
end
    
function Base.zeros(::Type{GrassmannLieAlgHorMatrix}, N::Integer, n::Integer)
    GrassmannLieAlgHorMatrix(
        zeros(N-n, n),
        N, 
        n
    )
end

function Base.rand(rng::Random.AbstractRNG, ::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    GrassmannLieAlgHorMatrix(rand(rng, T, N-n, n), N, n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{GrassmannLieAlgHorMatrix}, N::Integer, n::Integer)
    GrassmannLieAlgHorMatrix(rand(rng, N-n, n), N, n)
end

function Base.rand(::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where T
    rand(Random.default_rng(), GrassmannLieAlgHorMatrix{T}, N, n)
end

function Base.rand(::Type{GrassmannLieAlgHorMatrix}, N::Integer, n::Integer)
    rand(Random.default_rng(), GrassmannLieAlgHorMatrix, N, n)
end

function scalar_add(A::GrassmannLieAlgHorMatrix, δ::Real)
    GrassmannLieAlgHorMatrix(A.B .+ δ, A.N, A.n)
end

#define these functions more generally! (maybe make a fallback script!!)
function ⊙²(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(A.B.^2, A.N, A.n)
end
function RACᵉˡᵉ(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(sqrt.(A.B), A.N, A.n)
end
function /ᵉˡᵉ(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(A.B./B.B, A.N, A.n)
end 

function LinearAlgebra.mul!(C::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix, α::Real)
    mul!(C.B, A.B, α)
end
LinearAlgebra.mul!(C::GrassmannLieAlgHorMatrix, α::Real, A::GrassmannLieAlgHorMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::GrassmannLieAlgHorMatrix, α::Real) = mul!(C, C, α)