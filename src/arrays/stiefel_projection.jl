@doc raw"""
    StiefelProjection(backend, T, N, n)

Make a matrix of the form ``\begin{bmatrix} \mathbb{I} & \mathbb{O} \end{bmatrix}^T`` for a specific backend and data type.

An array that essentially does `vcat(I(n), zeros(N-n, n))` with GPU support. 

# Extend help

An instance of `StiefelProjection` should technically also belong to [`StiefelManifold`](@ref). 
"""
struct StiefelProjection{T, AT} <: AbstractMatrix{T}
    N::Integer
    n::Integer
    A::AT
    function StiefelProjection(backend, T::Type, N::Integer, n::Integer) 
        A = KernelAbstractions.zeros(backend, T, N, n)
        assign_ones_for_stiefel_projection! = assign_ones_for_stiefel_projection_kernel!(backend)
        assign_ones_for_stiefel_projection!(A, ndrange=n)
        new{T, typeof(A)}(N,n, A)
    end
end
    
@doc raw"""
    StiefelProjection(A::AbstractMatrix)

Extract necessary information from `A` and build an instance of `StiefelProjection`. 

Necessary information here referes to the backend, the data type and the size of the matrix.
"""
function StiefelProjection(A::AbstractMatrix{T}) where T 
    StiefelProjection(KernelAbstractions.get_backend(A), T, size(A)...)
end

@doc raw"""
    StiefelProjection(B::AbstractLieAlgHorMatrix)

Extract necessary information from `B` and build an instance of `StiefelProjection`. 

Necessary information here referes to the backend, the data type and the size of the matrix.

The size is queried through `B.N` and `B.n`.

# Examples

```jldoctest
using GeometricMachineLearning

B₁ = rand(StiefelLieAlgHorMatrix, 5, 2)
B₂ = rand(GrassmannLieAlgHorMatrix, 5, 2)
E = [1. 0.; 0. 1.; 0. 0.; 0. 0.; 0. 0.]

StiefelProjection(B₁) ≈ StiefelProjection(B₂) ≈ E 

# output

true
```
"""
function StiefelProjection(B::AbstractLieAlgHorMatrix{T}) where T 
    StiefelProjection(KernelAbstractions.get_backend(B), T, B.N, B.n)
end

@kernel function assign_ones_for_stiefel_projection_kernel!(A::AbstractArray{T}) where T
    i = @index(Global)
    A[i, i] = one(T)
end

StiefelProjection(N::Integer, n::Integer, T::Type=Float64) = StiefelProjection(CPU(), T, N, n)

StiefelProjection(T::Type, N::Integer, n::Integer) = StiefelProjection(N, n, T)

Base.size(E::StiefelProjection) = (E.N, E.n)
Base.getindex(E::StiefelProjection, i, j) = getindex(E.A, i, j)
Base.:+(E::StiefelProjection, A::AbstractMatrix) = E.A + A 
Base.:+(A::AbstractMatrix, E::StiefelProjection) = +(E, A)
Base.vcat(A::AbstractVecOrMat{T}, E::StiefelProjection{T}) where {T<:Number} = vcat(A, E.A)
Base.vcat(E::StiefelProjection{T}, A::AbstractVecOrMat{T}) where {T<:Number} = vcat(E.A, A)
Base.hcat(A::AbstractVecOrMat{T}, E::StiefelProjection{T}) where {T<:Number} = hcat(A, E.A)
Base.hcat(E::StiefelProjection{T}, A::AbstractVecOrMat{T}) where {T<:Number} = hcat(E.A, A)

function KernelAbstractions.get_backend(E::StiefelProjection)
    KernelAbstractions.get_backend(E.A)
end