"""
A `TriangularLowerMatrix` is a matrix
| 0  0 |
| S  0 |
Currently, it only implements a custom `mul!` method, exploiting this structure.

The first index is the row index, the second one the column index
"""

struct TriangularLowerMatrix{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int

    function TriangularLowerMatrix(S::AbstractVector,n::Int)
        @assert length(S) == n*(n-1)÷2
        new{eltype(S),typeof(S)}(S,n)
    end
    function TriangularLowerMatrix(S::AbstractMatrix)
        n = size(S)[1]
        @assert size(S)[2] == n
        S_vec = zeros(n*(n-1)÷2)
        #map the sub-diagonal elements to a vector 
        for i in 2:n
            S_vec[((i-1)*(i-2)÷2+1):(i*(i-1)÷2)] = S[i,1:(i-1)]
        end
        new{eltype(S),typeof(S_vec)}(S_vec,n)
    end

end 

    
Base.parent(A::TriangularLowerMatrix) = A.S
Base.size(A::TriangularLowerMatrix) = (A.n,A.n)
    
function LinearAlgebra.mul!(out::AbstractVector, A::TriangularLowerMatrix, z::AbstractVector)
    @assert length(out) == length(z) == A.n
    
    out[1] = 0

    for i in 1:(A.n-1)
        out[i+1] = z[1:i]'*A.S[(i*(i-1)÷2+1):(i*(i+1)÷2)]
    end
        
    return out
end

function LinearAlgebra.mul!(out::AbstractMatrix, A::TriangularLowerMatrix, Z::AbstractMatrix)
    @assert size(Z)[1] == A.n == size(out)[1]
    @assert size(out)[2] == size(Z)[2]
    for i in 1:(size(Z)[2]) out[:,i] = mul!(out[:,i],A,Z[:,i]) end
    return out 
end


function Base.getindex(A::TriangularLowerMatrix,i,j)
    if j ≥ i
        return zero(eltype(A))
    end
    return A.S[(i-2)*(i-1)÷2+j]
end

Base.:*(A::TriangularLowerMatrix, z::AbstractVector) = mul!(zeros(length(z)),A,z)
Base.:*(A::TriangularLowerMatrix, Z::AbstractMatrix) = mul!(zeros(A.n,size(Z)[2]),A,Z)