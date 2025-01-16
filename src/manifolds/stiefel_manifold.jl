@doc raw"""
    StiefelManifold <: Manifold

An implementation of the Stiefel manifold [hairer2006geometric](@cite). The Stiefel manifold is the collection of all matrices ``Y\in\mathbb{R}^{N\times{}n}`` whose columns are orthonormal, i.e. 

```math
    St(n, N) = \{Y: Y^TY = \mathbb{I}_n \}.
```

The Stiefel manifold can be shown to have manifold structure (as the name suggests) and this is heavily used in `GeometricMachineLearning`. It is further a compact space. 
More information can be found in the docstrings for [`rgrad(::StiefelManifold, ::AbstractMatrix)`](@ref) and [`metric(::StiefelManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref).
"""
mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
end

Base.:*(Y::StiefelManifold, B::AbstractMatrix) = Y.A*B
Base.:*(B::AbstractMatrix, Y::StiefelManifold) = B*Y.A

function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::AbstractMatrix) where {T, AT<:AbstractMatrix{T}}
    Y.parent.A'*B 
end

function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::StiefelManifold{T, AT}) where {T, AT<:AbstractMatrix{T}}
    Y.parent.A' * B.A 
end

function Base.:*(Y::Adjoint{T, ST}, B::ST) where {T, AT<:AbstractMatrix{T}, ST<:StiefelManifold{T, AT}}
    Y.parent.A' * B.A
end

@doc raw"""
    rgrad(Y::StiefelManifold, ∇L::AbstractMatrix)

Compute the Riemannian gradient for the Stiefel manifold at `Y` based on `∇L`.
    
Here ``Y\in{}St(N,n)`` and ``\nabla{}L\in\mathbb{R}^{N\times{}n}`` is the Euclidean gradient. 
    
The function computes the Riemannian gradient with respect to the canonical metric:
[`metric(::StiefelManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref).

The precise form of the mapping is: 
```math
\mathtt{rgrad}(Y, \nabla{}L) \mapsto \nabla{}L - Y(\nabla{}L)^TY
```

Note the property ``Y^T\mathtt{rgrad}(Y, \nabla{}L)\in\mathcal{S}_\mathrm{skew}(n).``

# Examples

```jldoctest
using GeometricMachineLearning

Y = StiefelManifold([1 0 ; 0 1 ; 0 0; 0 0])
Δ = [1 2; 3 4; 5 6; 7 8]
rgrad(Y, Δ)

# output

4×2 Matrix{Int64}:
 0  -1
 1   0
 5   6
 7   8
```
"""
function rgrad(Y::StiefelManifold, ∇L::AbstractMatrix)
    ∇L - Y.A * (∇L' * Y.A)
end

@doc raw"""
    metric(Y::StiefelManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)

Compute the dot product for `Δ₁` and `Δ₂` at `Y`.

This uses the canonical Riemannian metric for the Stiefel manifold:
```math 
g_Y: (\Delta_1, \Delta_2) \mapsto \mathrm{Tr}(\Delta_1^T(\mathbb{I} - \frac{1}{2}YY^T)\Delta_2).
```
"""
function metric(Y::StiefelManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    LinearAlgebra.tr(Δ₁'*(I - .5*Y.A*Y.A')*Δ₂)
end

function check(Y::StiefelManifold)
    norm(Y.A'*Y.A - I)
end

@doc raw"""
    global_section(Y::StiefelManifold)

Compute a matrix of size ``N\times(N-n)`` whose columns are orthogonal to the columns in `Y`.

This matrix is also called ``Y_\perp`` [absil2004riemannian, absil2008optimization, bendokat2020grassmann](@cite).

# Examples

```jldoctest
using GeometricMachineLearning
using GeometricMachineLearning: global_section
import Random

Random.seed!(123)

Y = StiefelManifold([1. 0.; 0. 1.; 0. 0.; 0. 0.])

round.(Matrix(global_section(Y)); digits = 3)

# output

4×2 Matrix{Float64}:
 0.0    -0.0
 0.0     0.0
 0.936  -0.353
 0.353   0.936
```

Further note that we convert the `QRCompactWYQ` object to a `Matrix` before we display it.

# Implementation

The implementation is done with a QR decomposition (`LinearAlgebra.qr!`). Internally we do: 

```julia
A = randn(N, N - n) # or the gpu equivalent
A = A - Y.A * (Y.A' * A)
qr!(A).Q
```
"""
function global_section(Y::StiefelManifold{T}) where T
    N, n = size(Y)
    backend = networkbackend(Y)
    A = KernelAbstractions.allocate(backend, T, N, N-n)
    randn!(A)
    A = A - Y.A * (Y.A' * A)
    typeof(Y.A)(qr!(A).Q)
end

@doc raw"""
    Ω(Y::StiefelManifold{T}, Δ::AbstractMatrix{T}) where T

Perform *canonical horizontal lift* for the Stiefel manifold:

```math
    \Delta \mapsto (\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T).
```

Internally this performs 

```julia
SkewSymMatrix(2 * (I(n) - .5 * Y * Y') * Δ * Y')
```

It uses [`SkewSymMatrix`](@ref) to save memory. 

# Examples 

```jldoctest
using GeometricMachineLearning
E = StiefelManifold(StiefelProjection(5, 2))
Δ = [0. -1.; 1. 0.; 2. 3.; 4. 5.; 6. 7.]
GeometricMachineLearning.Ω(E, Δ)

# output

5×5 SkewSymMatrix{Float64, Vector{Float64}}:
 0.0  -1.0  -2.0  -4.0  -6.0
 1.0   0.0  -3.0  -5.0  -7.0
 2.0   3.0   0.0  -0.0  -0.0
 4.0   5.0   0.0   0.0  -0.0
 6.0   7.0   0.0   0.0   0.0
```

Note that the output of `Ω` is a skew-symmetric matrix, i.e. an element of ``\mathfrak{g}``.
"""
function Ω(Y::StiefelManifold{T}, Δ::AbstractMatrix{T}) where T
    YY = Y * Y'
    SkewSymMatrix(2 * (one(YY) - T(.5) * Y * Y') * Δ * Y')
end

function Base.copyto!(A::StiefelManifold, B::StiefelManifold)
    A.A .= B.A
    nothing
end