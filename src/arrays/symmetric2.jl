"""
A `SymmetricMatrix` is a matrix
| a  S |
| S  b |

The first index is the row index, the second one the column index.

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
function Base.getindex(A::SymmetricMatrix,i,j)
    if i ≥ j
        return A.S[((i-1)*i)÷2+j]
    end
    return A.S[(j-1)*j÷2+i]
end


Base.parent(A::SymmetricMatrix) = A.S
Base.size(A::SymmetricMatrix) = (A.n,A.n)