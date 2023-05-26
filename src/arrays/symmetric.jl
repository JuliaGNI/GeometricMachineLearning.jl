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
    n::Int

    function SymmetricMatrix(S::AbstractVector,n::Int)
        @assert length(S) == n*(n+1)÷2
        new{eltype(S),typeof(S)}(S,n)
    end
    function SymmetricMatrix(S::AbstractMatrix)
        n = size(S)[1]
        @assert size(S)[2] == n
        S_vec = zeros(n*(n+1)÷2)
        #make the input symmetric if it isn't already
        S = .5*(S + S')
        #map the sub-diagonal elements to a vector 
        for i in 1:n
            S_vec[(i*(i-1)÷2+1):(i*(i+1)÷2)] = S[i,1:i]
        end
        new{eltype(S),typeof(S_vec)}(S_vec,n)
    end

end 

#implementing getindex automatically defines all matrix multiplications! (but probably not in the most efficient way)
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


