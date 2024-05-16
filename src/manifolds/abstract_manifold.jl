@doc raw"""
`rand` is implemented for manifolds that use the initialization of the [`StiefelManifold`](@ref) and the [`GrassmannManifold`](@ref) by default. 
"""
abstract type Manifold{T} <: AbstractMatrix{T} end

@kernel function assign_columns_kernel!(Y::AbstractMatrix{T}, A::AbstractMatrix{T}) where T
    i,j = @index(Global, NTuple)
    Y[i,j] = A[i,j]
end

function assign_columns(Q::AbstractMatrix{T}, N::Integer, n::Integer) where T
    backend = KernelAbstractions.get_backend(Q)
    Y = KernelAbstractions.allocate(backend, T, N, n)
    assign_columns! = assign_columns_kernel!(backend)
    assign_columns!(Y, Q, ndrange=size(Y))
    Y
end

# TODO: check the distribution this is coming from - related to the Haar measure ???
function Base.rand(::CPU, rng::Random.AbstractRNG, ::Type{MT}, N::Integer, n::Integer) where {T, MT<:Manifold{T}} 
    @assert N ≥ n 
    A = randn(rng, T, N, n)
    MT{typeof(A)}(assign_columns(typeof(A)(qr!(A).Q), N, n))
end

function Base.rand(backend::GPU, rng::Random.AbstractRNG, ::Type{MT}, N::Integer, n::Integer) where {T, MT<:Manifold{T}} 
    @assert N ≥ n 
    A = KernelAbstractions.allocate(backend, T, N, n)
    Random.randn!(rng, A)    
    MT{typeof(A)}(assign_columns(typeof(A)(qr!(A).Q), N, n))
end

function Base.rand(backend::CPU, rng::Random.AbstractRNG, ::Type{MT}, N::Integer, n::Integer) where MT <: Manifold
    rand(backend, rng, MT{Float64}, N, n)
end

function Base.rand(backend::GPU, rng::Random.AbstractRNG, ::Type{MT}, N::Integer, n::Integer) where MT <: Manifold
    rand(backend, rng, MT{Float32}, N, n)
end

function Base.rand(rng::Random.AbstractRNG, manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold
    rand(CPU(), rng, manifold_type, N, n)
end

function Base.rand(backend::KernelAbstractions.Backend, manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold 
    rand(backend, Random.default_rng(), manifold_type, N, n)
end

function Base.rand(manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold
    rand(Random.default_rng(), manifold_type, N, n)
end

Base.size(A::Manifold) = size(A.A)
Base.parent(A::Manifold) = A.A 
Base.getindex(A::Manifold, i::Int, j::Int) = A.A[i,j]