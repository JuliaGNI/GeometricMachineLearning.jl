"""
A `SymplecticLieAlgMatrix` is a matrix
| A  B  |
| C -A' |, where B and C are symmetric. 

The first index is the row index, the second one the column index.

TODO: Check how LinearAlgebra implements matrix multiplication!
"""

#-> AbstractVecOrMat!!
mutable struct SymplecticLieAlgMatrix{T, AT <: AbstractMatrix{T}, BT <: SymmetricMatrix{T}} <: AbstractMatrix{T}
    A::AT
    B::BT
    C::BT
    n::Int

    function SymplecticLieAlgMatrix(A::AbstractMatrix, B::SymmetricMatrix, C::SymmetricMatrix, n)
        @assert eltype(B) == eltype(C) == eltype(A)
        @assert B.n == C.n == n
        @assert size(A) == (n,n)
        new{eltype(B),typeof(A),typeof(B)}(A,B,C,n)
    end
    function SymplecticLieAlgMatrix(S::AbstractMatrix)
        n2 = size(S)[1]
        @assert iseven(n2)
        @assert size(S)[2] == n2
        n = n2÷2

        A = 0.5*(S[1:n,1:n] - S[(n+1):n2, (n+1):n2]')
        B = SymmetricMatrix(S[1:n,(n+1):n2])
        C = SymmetricMatrix(S[(n+1):n2,1:n])
  
        new{eltype(S), typeof(A), typeof(B)}(A, B, C, n)
    end

end 


#implementing getindex automatically defines all matrix multiplications! (but probably not in the most efficient way)
function Base.getindex(A::SymplecticLieAlgMatrix,i,j)
    n = A.n
    if i ≤ n && j ≤ n
        return A.A[i,j]
    end
    if i ≤ n
        return A.B[i, j-n]
    end
    if j ≤ n 
        return A.C[i-n, j]
    end
    return -A.A[j-n,i-n]
end


Base.parent(A::SymplecticLieAlgMatrix) = (A=A.A,B=A.B,C=A.C)
Base.size(A::SymplecticLieAlgMatrix) = (2*A.n,2*A.n)

function Base.:+(S₁::SymplecticLieAlgMatrix, S₂::SymplecticLieAlgMatrix) 
    @assert S₁.n == S₂.n  
    SymplecticLieAlgMatrix(
        S₁.A + S₂.A,
        S₁.B + S₂.B,
        S₁.C + S₂.C, 
        S₁.n
        )
end

function Base.:-(S₁::SymplecticLieAlgMatrix, S₂::SymplecticLieAlgMatrix) 
    @assert S₁.n == S₂.n  
    SymplecticLieAlgMatrix(
        S₁.A - S₂.A,
        S₁.B - S₂.B,
        S₁.C - S₂.C, 
        S₁.n
        )
end


#TODO: implement functions as in StiefelLieAlgHorMatrix