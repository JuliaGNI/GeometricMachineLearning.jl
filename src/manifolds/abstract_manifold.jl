@doc raw"""
    Manifold <: AbstractMatrix

A manifold in `GeometricMachineLearning` is a sutype of `AbstractMatrix`. All manifolds are matrix manifolds and therefore stored as matrices. More details can be found in the docstrings for the [`StiefelManifold`](@ref) and the [`GrassmannManifold`](@ref).
"""
abstract type Manifold{T} <: AbstractMatrix{T} end

@kernel function assign_columns_kernel!(Y::AbstractMatrix{T}, A::AbstractMatrix{T}) where T
    i,j = @index(Global, NTuple)
    Y[i,j] = A[i,j]
end

function assign_columns(Q::AbstractMatrix{T}, N::Integer, n::Integer) where T
    backend = networkbackend(Q)
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

function _round(Y::Manifold; kwargs...)
    typeof(Y)(round.(Y.A; kwargs...))
end

function Base.broadcast(operation, Y::Manifold)
    typeof(Y)(broadcast(operation, Y.A))
end

@doc raw"""
    rand(backend, manifold_type, N, n)

Draw random elements for a specific device.

# Examples

Random elements of the manifold can be allocated on GPU.  Call ...

```julia
rand(CUDABackend(), StiefelManifold{Float32}, N, n)
```

... for drawing elements on a `CUDA` device.
"""
function Base.rand(backend::KernelAbstractions.Backend, manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold 
    rand(backend, Random.default_rng(), manifold_type, N, n)
end

@doc raw"""
    rand(manifold_type, N, n)

Draw random elements from the Stiefel and the Grassmann manifold. 

Because both of these manifolds are compact spaces we can sample them uniformly [mezzadri2006generate](@cite).

# Examples
When we call ...

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: _round # hide
import Random
Random.seed!(123)

N, n = 5, 3
Y = rand(StiefelManifold{Float32}, N, n)
_round(Y; digits = 5) # hide

# output

5×3 StiefelManifold{Float32, Matrix{Float32}}:
 -0.27575   0.32991   0.77275
 -0.62485  -0.33224  -0.0686
 -0.69333   0.36724  -0.18988
 -0.09295  -0.73145   0.46064
  0.2102    0.33301   0.38717
```

... the sampling is done by first allocating a random matrix of size ``N\times{}n`` via `Y = randn(Float32, N, n)`.

We then perform a QR decomposition `Q, R = qr(Y)` with the `qr` function from the `LinearAlgebra` package (this is using Householder reflections internally). 

The final output are then the first `n` columns of the `Q` matrix. 
"""
function Base.rand(manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold
    rand(Random.default_rng(), manifold_type, N, n)
end

Base.size(A::Manifold) = size(A.A)
Base.parent(A::Manifold) = A.A 
Base.getindex(A::Manifold, i::Int, j::Int) = A.A[i,j]