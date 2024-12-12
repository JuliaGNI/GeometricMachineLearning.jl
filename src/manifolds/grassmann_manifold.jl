"""
    GrassmannManifold <: Manifold

The `GrassmannManifold` is based on the [`StiefelManifold`](@ref).
"""
mutable struct GrassmannManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
end

@doc raw"""
    rgrad(Y::GrassmannManifold, ∇L::AbstractMatrix)

Compute the Riemannian gradient for the Grassmann manifold at `Y` based on `∇L`.

Here ``Y`` is a representation of ``\mathrm{span}(Y)\in{}Gr(n, N)`` and ``\nabla{}L\in\mathbb{R}^{N\times{}n}`` is the Euclidean gradient. 

This gradient has the property that it is orthogonal to the space spanned by ``Y``.

The precise form of the mapping is: 
```math
\mathtt{rgrad}(Y, \nabla{}L) \mapsto \nabla{}L - YY^T\nabla{}L.
```

Note the property ``Y^T\mathrm{rgrad}(Y, \nabla{}L) = \mathbb{O}.``

Also see [`rgrad(::StiefelManifold, ::AbstractMatrix)`](@ref).

# Examples

```jldoctest
using GeometricMachineLearning

Y = GrassmannManifold([1 0 ; 0 1 ; 0 0; 0 0])
Δ = [1 2; 3 4; 5 6; 7 8]
rgrad(Y, Δ)

# output

4×2 Matrix{Int64}:
 0  0
 0  0
 5  6
 7  8
```
"""
function rgrad(Y::GrassmannManifold, ∇L::AbstractMatrix)
    ∇L - Y * (Y' * ∇L)
end

@doc raw"""
    metric(Y::GrassmannManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)

Compute the metric for vectors `Δ₁` and `Δ₂` at `Y`. 

The representation of the Grassmann manifold is realized as a *quotient space of the Stiefel manifold*. 

The metric for the Grassmann manifold is:

```math
g^{Gr}_Y(\Delta_1, \Delta_2) = g^{St}_Y(\Delta_1, \Delta_2) = \mathrm{Tr}(\Delta_1^T (\mathbb{I} - Y Y^T) \Delta_2) = \mathrm{Tr}(\Delta_1^T \Delta_2),
```
where we used that ``Y^T\Delta_i`` for ``i = 1, 2.``
"""
function metric(::GrassmannManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    LinearAlgebra.tr(Δ₁' * Δ₂)
end

@doc raw"""
    global_section(Y::GrassmannManifold)

Compute a matrix of size ``N\times(N-n)`` whose columns are orthogonal to the columns in `Y`.

The method `global_section` for the Grassmann manifold is equivalent to that for the [`StiefelManifold`](@ref) (we represent the Grassmann manifold as an embedding in the Stiefel manifold). 

See the documentation for [`global_section(Y::StiefelManifold{T}) where T`](@ref). 
"""
function global_section(Y::GrassmannManifold{T}) where T
    N, n = size(Y)
    backend = networkbackend(Y)
    A = KernelAbstractions.allocate(backend, T, N, N-n)
    randn!(A)
    A = A - Y.A * (Y.A' * A)
    typeof(Y.A)(qr!(A).Q)
end

@doc raw"""
    Ω(Y::GrassmannManifold{T}, Δ::AbstractMatrix{T}) where T

Perform the *canonical horizontal lift* for the Grassmann manifold:

```math
    \Delta \mapsto \Omega^{St}(\Delta),
```

where ``\Omega^{St}`` is the canonical horizontal lift for the Stiefel manifold.

```jldoctest
using GeometricMachineLearning
E = GrassmannManifold(StiefelProjection(5, 2))
Δ = [0. 0.; 0. 0.; 2. 3.; 4. 5.; 6. 7.]
GeometricMachineLearning.Ω(E, Δ)

# output

5×5 SkewSymMatrix{Float64, Vector{Float64}}:
 0.0  -0.0  -2.0  -4.0  -6.0
 0.0   0.0  -3.0  -5.0  -7.0
 2.0   3.0   0.0  -0.0  -0.0
 4.0   5.0   0.0   0.0  -0.0
 6.0   7.0   0.0   0.0   0.0
```
"""
function Ω(Y::GrassmannManifold{T}, Δ::AbstractMatrix{T}) where T
    YY = Y * Y'

    ΩSt = 2 * (one(YY) - T(.5) * Y * Y') * Δ * Y'
    # E = StiefelProjection(Y)
    # SkewSymMatrix(ΩSt - E * E' * ΩSt * E * E')
    SkewSymMatrix(ΩSt)
end