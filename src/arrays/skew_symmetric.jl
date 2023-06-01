"""
A `SkewSymMatrix` is a matrix
| 0 -S |
| S  0 |


If the constructor is called with a matrix as input it returns a symmetric matrix via the projection 
A ↦ .5*(A - Aᵀ). 
This is a projection defined via the canonical metric (A,B) ↦ tr(AᵀB).

The first index is the row index, the second one the column index.

TODO: Check how LinearAlgebra implements matrix multiplication!
"""

mutable struct SkewSymMatrix{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int

    function SkewSymMatrix(S::AbstractVector{T},n::Int) where {T}
        @assert length(S) == n*(n-1)÷2
        new{T,typeof(S)}(S,n)
    end
    function SkewSymMatrix(S::AbstractMatrix{T}) where {T}
        n = size(S, 1)
        @assert size(S, 2) == n
        S_vec = zeros(T, n*(n-1)÷2)
        #make the input skew-symmetric if it isn't already
        S = .5*(S - S')
        #map the sub-diagonal elements to a vector 
        for i in 2:n
            S_vec[((i-1)*(i-2)÷2+1):(i*(i-1)÷2)] = S[i,1:(i-1)]
        end
        new{T,typeof(S_vec)}(S_vec, n)
    end
end 

function Base.getindex(A::SkewSymMatrix, i::Int, j::Int)
    if j == i
        return zero(eltype(A))
    end
    if i > j
        return A.S[(i-2)*(i-1)÷2+j]
    end
    return - A.S[(j-2)*(j-1)÷2+i]
end


Base.parent(A::SkewSymMatrix) = A.S
Base.size(A::SkewSymMatrix) = (A.n,A.n)

function Base.:+(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n 
    SkewSymMatrix(A.S + B.S, A.n) 
end 

function add!(C::SkewSymMatrix, A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function Base.:-(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n 
    SkewSymMatrix(A.S - B.S, A.n) 
end 

function Base.:-(A::SkewSymMatrix)
    SkewSymMatrix(-A.S, A.n)
end

function Base.:*(A::SkewSymMatrix, α::Real)
    SkewSymMatrix(α*A.S, A.n)
end

Base.:*(α::Real, A::SkewSymMatrix) = A*α

function Base.zeros(::Type{SkewSymMatrix{T}}, n::Int) where T
    SkewSymMatrix(zeros(T, n*(n-1)÷2), n)
end
    
function Base.zeros(::Type{SkewSymMatrix}, n::Int)
    SkewSymMatrix(zeros(n*(n-1)÷2), n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SkewSymMatrix{T}}, n::Int) where T
    SkewSymMatrix(rand(rng, T, n*(n-1)÷2),n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SkewSymMatrix}, n::Int)
    SkewSymMatrix(rand(rng, n*(n-1)÷2), n)
end

#TODO: make defaults when no rng is specified!!! (prbabaly rng ← Random.default_rng())
function Base.rand(type::Type{SkewSymMatrix{T}}, n::Integer) where T
    rand(Random.default_rng(), type, n)
end

function Base.rand(type::Type{SkewSymMatrix}, n::Integer)
    rand(Random.default_rng(), type, n)
end

#these are Adam operations:
function scalar_add(A::SkewSymMatrix, δ::Real)
    SkewSymMatrix(A.S .+ δ, A.n)
end

#element-wise squares and square root (for Adam)
function ⊙²(A::SkewSymMatrix)
    SkewSymMatrix(A.S.^2, A.n)
end
function √ᵉˡᵉ(A::SkewSymMatrix)
    SkewSymMatrix(sqrt.(A.S), A.n)
end
function /ᵉˡᵉ(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n 
    SkewSymMatrix(A.S ./ B.S, A.n)
end

function LinearAlgebra.mul!(C::SkewSymMatrix, A::SkewSymMatrix, α::Real)
    mul!(C.S, A.S, α)
end
LinearAlgebra.mul!(C::SkewSymMatrix, α::Real, A::SkewSymMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::SkewSymMatrix, α::Real) = mul!(C, C, α)