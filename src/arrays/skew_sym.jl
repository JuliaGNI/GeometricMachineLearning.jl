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

    function SkewSymMatrix(S::AbstractVector,n::Int)
        @assert length(S) == n*(n-1)÷2
        new{eltype(S),typeof(S)}(S,n)
    end
    function SkewSymMatrix(S::AbstractMatrix)
        n = size(S)[1]
        @assert size(S)[2] == n
        S_vec = zeros(n*(n-1)÷2)
        #make the input skew-symmetric if it isn't already
        S = .5*(S - S')
        #map the sub-diagonal elements to a vector 
        for i in 2:n
            S_vec[((i-1)*(i-2)÷2+1):(i*(i-1)÷2)] = S[i,1:(i-1)]
        end
        new{eltype(S),typeof(S_vec)}(S_vec,n)
    end
    
    function SkewSymMatrix(n::Int)
        new{Float64, AbstractVector{Float64}}(randn(n*(n-1)÷2),n)
    end
end 

#somehow ranges (i.e. A[1:n, 1:n]) only work if I specify that i and j are indices!!!
#implementing getindex automatically defines all matrix multiplications! (but probably not in the most efficient way)
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

function Base.:-(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n 
    SkewSymMatrix(A.S - B.S, A.n) 
end 