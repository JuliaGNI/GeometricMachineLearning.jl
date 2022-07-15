
struct ZeroVector{T,N} <: AbstractVector{T}
    ZeroVector(T, N) = new{T,N}()
end

ZeroVector(x::AbstractVector) = ZeroVector(eltype(x), length(x))

Base.eltype(::ZeroVector{T,N}) where {T,N} = T
Base.length(::ZeroVector{T,N}) where {T,N} = N

function Base.getindex(::ZeroVector{T,N}, i) where {T,N}
    @assert i ≥ 1 && i ≤ N
    return zero(T)
end

function LinearAlgebra.mul!(out::AbstractVector, A::AbstractMatrix, z::ZeroVector)
    @assert length(axes(A,1)) == length(out)
    @assert length(axes(A,2)) == length(z)
    out .= 0
end
