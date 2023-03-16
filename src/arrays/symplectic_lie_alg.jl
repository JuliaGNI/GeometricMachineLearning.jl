"""
A `SymplecticLieAlgMatrix` is a matrix
| A  B  |
| C -A' |, where B and C are symmetric. 

The first index is the row index, the second one the column index.

TODO: Check how LinearAlgebra implements matrix multiplication!
"""

#-> AbstractVecOrMat!!
mutable struct SymplecticLieAlgMatrix{T, AT <: AbstractVector{T}, BT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    B::AT
    C::AT
    A::BT
    n::Int

    function SymplecticLieAlgMatrix(B::AbstractVector,C::AbstractVector,A::AbstractMatrix,n::Int)
        @assert eltype(B) == eltype(C) == eltype(A)
        @assert length(B) == n*(n+1)÷2
        @assert length(C) == n*(n+1)÷2
        @assert size(A) == (n,n)
        new{eltype(B),typeof(B),typeof(A)}(B,C,A,n)
    end
    function SymplecticLieAlgMatrix(S::AbstractMatrix)
        n = size(S)[1]
        @assert iseven(n)
        n ÷= 2
        @assert size(S)[2] == 2*n
        B_vec = zeros(n*(n+1)÷2)
        C_vec = zeros(n*(n+1)÷2)
        A_mat = zeros(n,n)
        #map the sub-diagonal elements to a vector (with an additional symmetrization)
        for i in 1:n
            B_vec[(i*(i-1)÷2+1):(i*(i+1)÷2)] = 0.5*(S[1:n,(n+1):2*n][i,1:i] + S[1:n,(n+1):2*n][1:i,i])
            C_vec[(i*(i-1)÷2+1):(i*(i+1)÷2)] = 0.5*(S[(n+1):2*n,1:n][i,1:i] + S[(n+1):2*n,1:n][1:i,i])
        end
        A_mat = S[1:n,1:n]
        new{eltype(S),typeof(B_vec),typeof(A_mat)}(B_vec,C_vec,A_mat,n)
    end

end 


#implementing getindex automatically defines all matrix multiplications! (but probably not in the most efficient way)
function Base.getindex(A::SymplecticLieAlgMatrix,i,j)
    if i ≤ A.n && j ≤ A.n
        return A.A[i,j]
    end
    if i ≤ A.n 
        j = j-A.n   
        if i ≥ j
            return A.B[((i-1)*i)÷2+j]
        end
        return A.B[(j-1)*j÷2+i]
    end
    if j ≤ A.n 
        i = i-A.n   
        if i ≥ j
            return A.C[((i-1)*i)÷2+j]
        end
        return A.C[(j-1)*j÷2+i]
    end
    i = i - A.n 
    j = j - A.n 
    return -A.A[j,i]
end


Base.parent(A::SymplecticLieAlgMatrix) = (A=A.A,B=A.B,C=A.C)
Base.size(A::SymplecticLieAlgMatrix) = (2*A.n,2*A.n)

#function Base.:./(A::SymplecticLieAlgMatrix,B::SymplecticLieAlgMatrix)
function Adam_div(A::SymplecticLieAlgMatrix,B::SymplecticLieAlgMatrix)
    @assert A.n == B.n
    SymplecticLieAlgMatrix(A.B./B.B,A.C./B.C,A.A./B.A,A.n)
end

⊙(A::SymplecticLieAlgMatrix) = SymplecticLieAlgMatrix(A.B.^2,A.C.^2,A.A.^2,A.n)

Base.:√(A::SymplecticLieAlgMatrix) = SymplecticLieAlgMatrix(sqrt.(A.B),sqrt.(A.C),sqrt.(A.A),A.n)

