@doc raw"""
An implementation of the Stiefel manifold. It has various convenience functions associated with it:
- check 
- rand 
- rgrad
- metric
"""
mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
end

Base.:*(Y::StiefelManifold, B::AbstractMatrix) = Y.A*B
Base.:*(B::AbstractMatrix, Y::StiefelManifold) = B*Y.A

function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::AbstractMatrix) where {T, AT<:AbstractMatrix{T}}
    Y.parent.A'*B 
end

function Base.:*(Y::Adjoint{T, ST}, B::ST) where {T, AT<:AbstractMatrix{T}, ST<:StiefelManifold{T, AT}}
    Y.parent.A' * B.A
end

@doc raw"""
Computes the Riemannian gradient for the Stiefel manifold given an element ``Y\in{}St(N,n)`` and a matrix ``\nabla{}L\in\mathbb{R}^{N\times{}n}`` (the Euclidean gradient). It computes the Riemannian gradient with respect to the canonical metric (see the documentation for the function `metric` for an explanation of this).
The precise form of the mapping is: 
```math
\mathtt{rgrad}(Y, \nabla{}L) \mapsto \nabla{}L - Y(\nabla{}L)^TY
```
It is called with inputs:
- `Y::StiefelManifold`
- `e_grad::AbstractMatrix`: i.e. the Euclidean gradient (what was called ``\nabla{}L``) above.
"""
function rgrad(Y::StiefelManifold, e_grad::AbstractMatrix)
    e_grad - Y.A * (e_grad' * Y.A)
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

function global_section(Y::StiefelManifold{T}) where T
    N, n = size(Y)
    backend = KernelAbstractions.get_backend(Y)
    A = KernelAbstractions.allocate(backend, T, N, N-n)
    randn!(A)
    A = A - Y.A * (Y.A' * A)
    qr!(A).Q
end