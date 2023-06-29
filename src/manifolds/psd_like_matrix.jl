mutable struct PsdLike{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
    function PsdLike(A::AbstractMatrix)
        @assert size(A,1) ≥ size(A,2)
        new{eltype(A), typeof(A)}(A)
    end
end

Base.size(A::PsdLike) = 2 .*size(A.A)

function Base.getindex(A::PsdLike{T}, i::Integer, j::Integer) where {T}
    N, n = size(A).÷2
    if j < n 
        if i < N
            return A.A[i, j]
        end
        return T(0.)
    end
    if i < N 
        return T(0.)
    end
    return A.A[i-N, j-n]
end


function Base.rand(rng::Random.AbstractRNG, ::Type{PsdLike{T}}, N2::Integer, n2::Integer) where {T}
    @assert N2 ≥ n2
    @assert iseven(N2)
    @assert iseven(n2)
    A = randn(rng, T, N2÷2, n2÷2)
    PsdLike(qr!(A).Q[:, 1:(n2÷2)])
end

function Base.rand(rng::Random.AbstractRNG, ::Type{PsdLike}, N2::Integer, n2::Integer)
    @assert N2 ≥ n2
    @assert iseven(N2)
    @assert iseven(n2)
    A = randn(rng, N2÷2, n2÷2)
    PsdLike(qr!(A).Q[:, 1:(n2÷2)])
end

function Base.rand(::Type{PsdLike{T}}, N2::Integer, n2::Integer) where {T}
    @assert N2 ≥ n2
    @assert iseven(N2)
    @assert iseven(n2)
    A = randn(T, N2÷2, n2÷2)
    PsdLike(qr!(A).Q[:, 1:(n2÷2)])
end

function Base.rand(::Type{PsdLike}, N2::Integer, n2::Integer)
    @assert N2 ≥ n2
    @assert iseven(N2)
    @assert iseven(n2)
    A = randn(N2÷2, n2÷2)
    PsdLike(qr!(A).Q[:, 1:(n2÷2)])
end

function Base.rand(::TrivialInitRNG, ::Type{PsdLike{T}}, N2::Integer, n2::Integer) where T
    @assert N2 ≥ n2
    @assert iseven(N2)
    @assert iseven(n2)
    N, n = N2÷2, n2÷2 
    zeros(StiefelLieAlgHorMatrix{T}, N, n)
end

function Base.rand(::TrivialInitRNG, ::Type{PsdLike}, N::Integer, n::Integer)
    @assert N2 ≥ n2
    @assert iseven(N2)
    @assert iseven(n2)
    N, n = N2÷2, n2÷2 
    zeros(StiefelLieAlgHorMatrix, N, n)
end