@doc raw"""
An array that essentially does `vcat(I(n), zeros(N-n, n))` with GPU support. It has **three inner constructors**. The **first one** is called with the following arguments: 
1. `backend`: backends as supported by `KernelAbstractions`.
2. `T::Type`
3. `N::Integer`
4. `n::Integer`

The **second constructor** is called by supplying a matrix as input. The constructor will then extract the backend, the type and the dimensions of that matrix. 

The **third constructor** is called by supplying an instance of `StiefelLieAlgHorMatrix`.  

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

    function StiefelProjection(A::AbstractMatrix{T}) where T 
        StiefelProjection(KernelAbstractions.get_backend(A), T, size(A)...)
    end

    function StiefelProjection(B::StiefelLieAlgHorMatrix{T}) where T 
        StiefelProjection(KernelAbstractions.get_backend(B), T, B.N, B.n)
    end
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