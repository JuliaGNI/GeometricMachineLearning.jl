@doc raw"""
    LowerTriangular(S::AbstractVector, n::Int)

Build a lower-triangular matrix from a vector.

A lower-triangular matrix is an ``n\times{}n`` matrix that has zeros on the diagonal and on the upper triangular.

The data are stored in a vector ``S`` similarly to other matrices. See [`UpperTriangular`](@ref), [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref).

The struct two fields: `S` and `n`. The first stores all the entries of the matrix in a sparse fashion (in a vector) and the second is the dimension ``n`` for ``A\in\mathbb{R}^{n\times{}n}``.

# Examples 
```jldoctest
using GeometricMachineLearning
S = [1, 2, 3, 4, 5, 6]
LowerTriangular(S, 4)

# output

4×4 LowerTriangular{Int64, Vector{Int64}}:
 0  0  0  0
 1  0  0  0
 2  3  0  0
 4  5  6  0
```
"""
mutable struct LowerTriangular{T, AT <: AbstractVector{T}} <: AbstractTriangular{T}
    S::AT
    n::Int
end 

@doc raw"""
    LowerTriangular(A::AbstractMatrix)

Build a lower-triangular matrix from a matrix.

This is done by taking the lower left of that matrix.

# Examples 
```jldoctest
using GeometricMachineLearning
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
LowerTriangular(M)

# output

4×4 LowerTriangular{Int64, Vector{Int64}}:
  0   0   0  0
  5   0   0  0
  9  10   0  0
 13  14  15  0
```
"""
function LowerTriangular(S::AbstractMatrix{T}) where {T}
    n = size(S, 1)
    @assert size(S, 2) == n
    S_vec = map_to_lo(S)
    LowerTriangular(S_vec, n)
end

function Base.getindex(A::LowerTriangular{T}, i::Int, j::Int) where T
    if j == i
        return zero(T)
    end
    if i > j
        return A.S[(i-2) * (i-1) ÷ 2 + j]
    end
    return zero(T) 
end

@kernel function lo_mat_mul_kernel!(C::AbstractMatrix{T}, S::AbstractVector{T}, B::AbstractMatrix{T}, n) where T
    i, j = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k = 1:(i-1)
        tmp_sum +=  S[(i-2)*(i-1)÷2+k] * B[k, j]
    end
    C[i,j] = tmp_sum
end

function map_to_lo(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    @assert size(A, 2) == n 
    backend = KernelAbstractions.get_backend(A)
    S = KernelAbstractions.zeros(backend, T, n * (n - 1) ÷ 2)
    assign_Skew_val! = assign_Skew_val_kernel!(backend)
    for i in 2:n
        assign_Skew_val!(S, A, i, ndrange = (i - 1))
    end
    S
end

# define routines for generalizing ChainRulesCore to LowerTriangular 
ChainRulesCore.ProjectTo(A::AT) where AT <: LowerTriangular = ProjectTo{AT}(; triang = ProjectTo(A.S))
(project::ProjectTo{<:LowerTriangular})(dA::AbstractMatrix) = LowerTriangular(project.triang(map_to_lo(dA)), size(dA, 2))
(project::ProjectTo{<:LowerTriangular})(dA::LowerTriangular) = LowerTriangular(project.triang(dA.S), dA.n)