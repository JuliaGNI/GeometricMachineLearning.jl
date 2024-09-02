@doc raw"""
    UpperTriangular(S::AbstractVector, n::Int)

Build an upper-triangular matrix from a vector.

An upper-triangular matrix is an ``n\times{}n`` matrix that has zeros on the diagonal and on the lower triangular.

The data are stored in a vector ``S`` similarly to other matrices. See [`LowerTriangular`](@ref), [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref).

The struct two fields: `S` and `n`. The first stores all the entries of the matrix in a sparse fashion (in a vector) and the second is the dimension ``n`` for ``A\in\mathbb{R}^{n\times{}n}``.

# Examples 
```jldoctest
using GeometricMachineLearning
S = [1, 2, 3, 4, 5, 6]
UpperTriangular(S, 4)

# output

4×4 UpperTriangular{Int64, Vector{Int64}}:
 0  1  2  4
 0  0  3  5
 0  0  0  6
 0  0  0  0
```
"""
mutable struct UpperTriangular{T, AT <: AbstractVector{T}} <: AbstractTriangular{T}
    S::AT
    n::Int
end 

@doc raw"""
    UpperTriangular(A::AbstractMatrix)

Build an upper-triangular matrix from a matrix.

This is done by taking the upper right of that matrix.

# Examples 
```jldoctest
using GeometricMachineLearning
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
UpperTriangular(M)

# output

4×4 UpperTriangular{Int64, Vector{Int64}}:
 0  2  3   4
 0  0  7   8
 0  0  0  12
 0  0  0   0
```
"""
function UpperTriangular(S::AbstractMatrix{T}) where {T}
    n = size(S, 1)
    @assert size(S, 2) == n
    S_vec = map_to_up(S)
    UpperTriangular(S_vec, n)
end

function Base.getindex(A::UpperTriangular{T}, i::Int, j::Int) where T
    if j == i
        return zero(T)
    end
    if j > i
        return A.S[(j-2) * (j-1) ÷ 2 + i]
    end
    return zero(T) 
end

@kernel function up_mat_mul_kernel!(C::AbstractMatrix{T}, S::AbstractVector{T}, B::AbstractMatrix{T}, n) where T
    i, j = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k = (i + 1):n 
        tmp_sum += S[(k - 2) * (k - 1) ÷ 2 + i] * B[k, j]
    end
    C[i, j] = tmp_sum
end

function map_to_up(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    @assert size(A, 2) == n 
    backend = KernelAbstractions.get_backend(A)
    S = KernelAbstractions.zeros(backend, T, n * (n - 1) ÷ 2)
    assign_Skew_val! = assign_Skew_val_kernel!(backend)
    for i in 2:n
        assign_Skew_val!(S, A', i, ndrange = (i - 1))
    end
    S
end


# define routines for generalizing ChainRulesCore to UpperTriangular 
ChainRulesCore.ProjectTo(A::AT) where AT <: UpperTriangular = ProjectTo{AT}(; triang = ProjectTo(A.S))
(project::ProjectTo{<:UpperTriangular})(dA::AbstractMatrix) = UpperTriangular(project.triang(map_to_up(dA)), size(dA, 2))
(project::ProjectTo{<:UpperTriangular})(dA::UpperTriangular) = UpperTriangular(project.triang(dA.S), dA.n)

function Base.adjoint(A::LowerTriangular)
    UpperTriangular(A.S, A.n)
end

function Base.adjoint(A::UpperTriangular)
    LowerTriangular(A.S, A.n)
end