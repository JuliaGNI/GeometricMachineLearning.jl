@doc raw"""
A manifold in `GeometricMachineLearning` is a sutype of `AbstractMatrix`. All manifolds are matrix manifolds and therefore stored as matrices. More details can be found in the docstrings for the [`StiefelManifold`](@ref) and the [`GrassmannManifold`](@ref).
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

@doc raw"""
    rand(backend::KernelAbstractions.Backend, manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold)

Draw random elements for a specific device.

# Examples
```jldoctest
using GeometricMachineLearning
import Random
Random.seed!(123)

N, n = 5, 3
rand(CPU(), StiefelManifold{Float32}, N, n)

# output

5×3 StiefelManifold{Float32, Matrix{Float32}}:
 -0.275746    0.329913   0.772753
 -0.624851   -0.332242  -0.0685992
 -0.693326    0.367239  -0.189882
 -0.0929493  -0.731446   0.460639
  0.210203    0.333008   0.387173
```

Random elements of the manifold can also be allocated on GPU, via e.g. ...

```julia
rand(CUDABackend(), StiefelManifold{Float32}, N, n)
```

... for drawing elements on a `CUDA` device.
"""
function Base.rand(backend::KernelAbstractions.Backend, manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold 
    rand(backend, Random.default_rng(), manifold_type, N, n)
end

@doc raw"""
    rand(manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold

Draw random elements from the Stiefel and the Grassmann manifold. 

Because both of these manifolds are compact spaces we can sample them uniformly [mezzadri2006generate](@cite).

# Examples
When we call ...

```jldoctest
using GeometricMachineLearning
import Random
Random.seed!(123)

N, n = 5, 3
rand(StiefelManifold{Float32}, N, n)

# output

5×3 StiefelManifold{Float32, Matrix{Float32}}:
 -0.275746    0.329913   0.772753
 -0.624851   -0.332242  -0.0685992
 -0.693326    0.367239  -0.189882
 -0.0929493  -0.731446   0.460639
  0.210203    0.333008   0.387173
```

... the sampling is done by first allocating a random matrix of size ``N\times{}n`` via `Y = randn(Float32, N, n)`. We then perform a QR decomposition `Q, R = qr(Y)` with the `qr` function from the `LinearAlgebra` package (this is using Householder reflections internally). 
The final output are then the first `n` columns of the `Q` matrix. 
"""
function Base.rand(manifold_type::Type{MT}, N::Integer, n::Integer) where MT <: Manifold
    rand(Random.default_rng(), manifold_type, N, n)
end

Base.size(A::Manifold) = size(A.A)
Base.parent(A::Manifold) = A.A 
Base.getindex(A::Manifold, i::Int, j::Int) = A.A[i,j]