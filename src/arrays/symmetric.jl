"""
A `SymmetricMatrix` is a matrix
| a  S |
| S  b |

The first index is the row index, the second one the column index.

If the constructor is called with a matrix as input it returns a symmetric matrix via the projection 
A ↦ .5*(A + Aᵀ). 
This is a projection defined via the canonical metric (A,B) ↦ tr(AᵀB).

TODO: Overload Adjoint operation for SymmetricMatrix!! (Aᵀ = A)
TODO: Check how LinearAlgebra implements matrix multiplication!
"""

mutable struct SymmetricMatrix{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Integer

    function SymmetricMatrix(S::AbstractVector,n::Integer)
        @assert length(S) == n*(n+1)÷2
        new{eltype(S),typeof(S)}(S, n)
    end
    function SymmetricMatrix(S::AbstractMatrix{T}) where {T}
        n = size(S)[1]
        @assert size(S)[2] == n
        S_vec = zeros(T, n*(n+1)÷2)
        #make the input symmetric if it isn't already
        S = T(.5)*(S + S')
        #this is disgusting and should be removed! Here because indexing for GPUs not supported.
        S_cpu = Matrix{T}(S)
        #map the sub-diagonal elements to a vector 
        for i in 1:n
            S_vec[(i*(i-1)÷2+1):(i*(i+1)÷2)] = S_cpu[i,1:i]
        end
        S_vec₂ = Base.typename(typeof(S)).wrapper{eltype(S), 1}(S_vec)
        new{T, typeof(S_vec₂)}(S_vec₂, n)
    end
end 


function Base.getindex(A::SymmetricMatrix,i::Int,j::Int)
    if i ≥ j
        return A.S[((i-1)*i)÷2+j]
    end
    return A.S[(j-1)*j÷2+i]
end


Base.parent(A::SymmetricMatrix) = A.S
Base.size(A::SymmetricMatrix) = (A.n,A.n)

function Base.:+(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n 
    SymmetricMatrix(A.S + B.S, A.n) 
end 

function add!(C::SymmetricMatrix, A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function Base.:-(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n 
    SymmetricMatrix(A.S - B.S, A.n) 
end 

function Base.setindex!(A::SymmetricMatrix{T},a::T,i::Int,j::Int) where{T}
    if i ≥ j
        A.S[(i-1)*i÷2+j] = a
    else
        A.S[(j-1)*j÷2+i] = a
    end
end

function Base.:-(A::SymmetricMatrix)
    SymmetricMatrix(-A.S, A.n)
end

function Base.:*(A::SymmetricMatrix, α::Real)
    SymmetricMatrix(α*A.S, A.n)
end

Base.:*(α::Real, A::SymmetricMatrix) = A*α

function Base.zeros(::Type{SymmetricMatrix{T}}, n::Int) where T
    SymmetricMatrix(zeros(T, n*(n+1)÷2), n)
end
    
function Base.zeros(::Type{SymmetricMatrix}, n::Int)
    SymmetricMatrix(zeros(n*(n+1)÷2), n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymmetricMatrix{T}}, n::Int) where T
    SymmetricMatrix(rand(rng, T, n*(n+1)÷2),n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymmetricMatrix}, n::Int)
    SymmetricMatrix(rand(rng, n*(n+1)÷2), n)
end

#TODO: make defaults when no rng is specified!!! (prbabaly rng ← Random.default_rng())
function Base.rand(type::Type{SymmetricMatrix{T}}, n::Integer) where T
    rand(Random.default_rng(), type, n)
end

function Base.rand(type::Type{SymmetricMatrix}, n::Integer)
    rand(Random.default_rng(), type, n)
end

#these are Adam operations:
function scalar_add(A::SymmetricMatrix, δ::Real)
    SymmetricMatrix(A.S .+ δ, A.n)
end

#element-wise squares and square root (for Adam)
function ⊙²(A::SymmetricMatrix)
    SymmetricMatrix(A.S.^2, A.n)
end
function √ᵉˡᵉ(A::SymmetricMatrix)
    SymmetricMatrix(sqrt.(A.S), A.n)
end
function /ᵉˡᵉ(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n 
    SymmetricMatrix(A.S ./ B.S, A.n)
end

function LinearAlgebra.mul!(C::SymmetricMatrix, A::SymmetricMatrix, α::Real)
    mul!(C.S, A.S, α)
end
LinearAlgebra.mul!(C::SymmetricMatrix, α::Real, A::SymmetricMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::SymmetricMatrix, α::Real) = mul!(C, C, α)

#=
function CUDA.cu(A::SymmetricMatrix)
    SymmetricMatrix(CUDA.cu(A.S), A.n)
end
=#