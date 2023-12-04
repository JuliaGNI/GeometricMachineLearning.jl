@doc raw"""
An implementation of the Stiefel manifold. It has various convenience functions associated with it:
- check 
- rand 
- rgrad
- metric
"""
mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
    function StiefelManifold(A::AbstractMatrix)
        @assert size(A)[1] ≥ size(A)[2]
        new{eltype(A), typeof(A)}(A)
    end
end

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
function Base.rand(rng::Random.AbstractRNG, ::Type{StiefelManifold{T}}, N::Integer, n::Integer) where T
    @assert N ≥ n
    A = randn(rng, T, N, n)
    StiefelManifold(assign_columns(typeof(A)(qr!(A).Q), N, n))
end

function Base.rand(rng::Random.AbstractRNG, ::Type{StiefelManifold}, N::Integer, n::Integer)
    rand(rng, StiefelManifold{Float64}, N, n)
end

function Base.rand(manifold_type::Type{StiefelManifold{T}}, N::Integer, n::Integer) where T
    rand(Random.default_rng(), manifold_type, N, n)
end

function Base.rand(::Type{StiefelManifold}, N::Integer, n::Integer)
    rand(StiefelManifold{Float64}, N, n)
end

function Base.rand(backend::KernelAbstractions.Backend, rng::Random.AbstractRNG, ::Type{StiefelManifold{T}}, N::Integer, n::Integer) where T 
    @assert N ≥ n 
    A = KernelAbstractions.allocate(backend, T, N, n)
    Random.rand!(rng, A)
    StiefelManifold(assign_columns(typeof(A)(qr!(A).Q), N, n))
end

function Base.rand(backend::KernelAbstractions.Backend, manifold_type::Type{StiefelManifold{T}}, N::Integer, n::Integer) where T 
    rand(backend, Random.default_rng(), manifold_type, N, n)
end

Base.:*(Y::StiefelManifold, B::AbstractMatrix) = Y.A*B
Base.:*(B::AbstractMatrix, Y::StiefelManifold) = B*Y.A

function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::AbstractMatrix) where {T, AT<:AbstractMatrix{T}}
    Y.parent.A'*B 
end

@doc raw"""
Computes the Riemannian gradient for the Stiefel manifold given an element ``Y\in{}St(N,n)`` and a matrix ``\nabla{}L\in\mahbb{R}^{N\times{}n}`` (the Euclidean gradient). It computes the Riemannian gradient with respect to the canonical metric (see the documentation for the function `metric` for an explanation of this).
The precise form of the mapping is: 
```math
\mathtt{rgrad}(Y, \nabla{}L) \mapsto \nabla{}L - Y(\nabla{}L)^TY
```
It is called with inputs:
- `Y::StiefelManifold`
- `e_grad::AbstractMatrix`: i.e. the Euclidean gradient (what was called ``\nabla{}L``) above.
"""
function rgrad(Y::StiefelManifold, e_grad::AbstractMatrix)
    e_grad - Y.A*(e_grad'*Y.A)
end

@doc raw"""
Implements the canonical Riemannian metric for the Stiefel manifold:
```math 
g_Y: (\Delta_1, \Delta_2) \mapsto \mathrm{tr}(\Delta_1^T(\mathbb{I} - \frac{1}{2}YY^T)\Delta_2).
```
It is called with: 
- `Y::StiefelManifold`
- `Δ₁::AbstractMatrix`
- `Δ₂::AbstractMatrix``
"""
function metric(Y::StiefelManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    LinearAlgebra.tr(Δ₁'*(I - .5*Y.A*Y.A')*Δ₂)
end

function check(Y::StiefelManifold)
    norm(Y.A'*Y.A - I)
end

function global_section(Y::StiefelManifold)
    N, n = size(Y)
    A = typeof(Y.A)(randn(eltype(Y), N, N-n))
    A = A - Y.A*Y.A'*A
    qr!(A).Q
end