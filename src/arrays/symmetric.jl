"""
A `SymmetricMatrix` is a matrix W + W'
Currently, it only implements custom `mul!` and `*` methods, exploiting this structure.
"""
struct SymmetricMatrix{T, AT <: AbstractMatrix{T}} <: AbstractMatrix{T}
    S::AT

    function SymmetricMatrix(S::AbstractMatrix)
        @assert length(axes(S,1)) == length(axes(S,2))
        new{eltype(S), typeof(S)}(S)
    end
end

#getindex(A::SymmetricMatrix,inds...) = getindex(A.S,inds...)


Base.parent(A::SymmetricMatrix) = A.S
Base.size(A::SymmetricMatrix) = size(parent(A))

function LinearAlgebra.mul!(out::AbstractVector, A::SymmetricMatrix, z::AbstractVector)
    @assert length(out) == length(z) == length(axes(A.S,1)) == length(axes(A.S,2))

    mul!(out, A.S, z)
    mul!(out, A.S', z, 1, 1)

    return out
end

Base.getindex(A::SymmetricMatrix, i, j) = A.S[i,j] + A.S[j,i]

Base.:*(A::SymmetricMatrix, B::AbstractVector) = A.S * B + A.S' * B
Base.:*(A::AbstractVector, B::SymmetricMatrix) = A * B.S + A * B.S'

Base.:*(A::SymmetricMatrix, B::AbstractVecOrMat) = A.S * B + A.S' * B
Base.:*(A::AbstractVecOrMat, B::SymmetricMatrix) = A * B.S + A * B.S'




