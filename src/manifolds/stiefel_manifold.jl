@doc raw"""
An implementation of the Stiefel manifold [hairer2006geometric](@cite). The Stiefel manifold is the collection of all matrices ``Y\in\mathbb{R}^{N\times{}n}`` whose columns are orthonormal, i.e. 

```math
    St(n, N) = \{Y: Y^TY = \mathbb{I}_n \}.
```

The Stiefel manifold can be shown to have manifold structure (as the name suggests) and this is heavily used in `GeometricMachineLearning`. It is further a compact space. 
More information can be found in the docstrings for `rgrad(::StiefelManifold, ::AbstractMatrix)`` and `metric(::StiefelManifold, ::AbstractMatrix, ::AbstractMatrix)`.
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
- `Δ₂::AbstractMatrix`
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

@doc raw"""
Implements the *canonical horizontal lift* for the Stiefel manifold:

```math
    (\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T).
```

Internally this performs 

```julia
SkewSymMatrix(2 * (I(n) - .5 * Y * Y') * Δ * Y')
```

to save memory. 
"""
function Ω(Y::StiefelManifold{T}, Δ::AbstractMatrix{T}) where T
    YY = Y * Y'
    SkewSymMatrix(2 * (one(YY) - .5 * Y * Y') * Δ * Y')
end