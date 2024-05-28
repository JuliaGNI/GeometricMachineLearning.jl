@doc raw"""
    StiefelProjection(backend, T, N, n)

Make a matrix of the form ``\begin{pmatrix} \mathbb{I} & \mathbb{O} \end{pmatrix}^T`` for a specific backend and data type.

An array that essentially does `vcat(I(n), zeros(N-n, n))` with GPU support. 

# Extend help

Technically this should be a subtype of `StiefelManifold`. 
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
    StiefelProjection(B::StiefelLieAlgHorMatrix)

Extract necessary information from `B` and build an instance of `StiefelProjection`. 

Necessary information here referes to the backend, the data type and the size of the matrix.

The size is queried through `B.N` and `B.n`.
"""
function StiefelProjection(B::StiefelLieAlgHorMatrix{T}) where T 
    StiefelProjection(KernelAbstractions.get_backend(B), T, B.N, B.n)
end

@kernel function assign_ones_for_stiefel_projection_kernel!(A::AbstractArray{T}) where T
    i = @index(Global)
    A[i, i] = one(T)
end

"""
Outer constructor for `StiefelProjection`. This works with two integers as input and optionally the type.
"""
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