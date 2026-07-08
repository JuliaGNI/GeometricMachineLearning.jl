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

function Base.copyto!(A::GrassmannManifold, B::GrassmannManifold)
    A.A .= B.A
    nothing
end

Base.copy(A::GrassmannManifold) = GrassmannManifold(copy(A.A))
Base.similar(A::GrassmannManifold) = GrassmannManifold(similar(A.A))

function GeometricOptimizers.global_rep(
    λY::GeometricOptimizers.GlobalSection{T, <:GrassmannManifold{T}},
    Δ::AbstractMatrix{T}
) where T
    N, n = size(λY.Y)
    GrassmannLieAlgHorMatrix(
        λY.λ' * Δ,
        N, n
    )
end

function GeometricOptimizers.update_section!(
    Λᵗ::GeometricOptimizers.GlobalSection{T, <:GrassmannManifold{T}},
    Λ⁽ᵗ⁻¹⁾::GeometricOptimizers.GlobalSection{T, <:GrassmannManifold{T}},
    B⁽ᵗ⁻¹⁾::AbstractMatrix{T},
    retraction
) where T
    N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
    expB = retraction(B⁽ᵗ⁻¹⁾)
    expB.A .= Λ⁽ᵗ⁻¹⁾.Y.A * expB.A[1:n, :] .+ Λ⁽ᵗ⁻¹⁾.λ * expB.A[(n+1):N, :]
    Λᵗ.Y.A .= @view expB.A[:, 1:n]
    Λᵗ.λ .= @view expB.A[:, (n+1):N]
    nothing
end

function GeometricOptimizers.cayley(B::GrassmannLieAlgHorMatrix{T}) where T
    backend = networkbackend(B)
    E = StiefelProjection(B)
    𝕆 = KernelAbstractions.zeros(backend, T, B.n, B.n)
    𝕀_small = one(𝕆)
    𝕀_small2 = hcat(vcat(𝕀_small, 𝕆), vcat(𝕆, 𝕀_small))
    𝕀_big = one(B)
    B̂ = hcat(vcat(𝕆, B.B), E)
    B̄ = hcat(vcat(𝕀_small, 𝕆), vcat(zero(B.B'), -B.B'))'
    GrassmannManifold((𝕀_big + T(0.5) * B̂ * inv(𝕀_small2 - T(0.5) * B̄' * B̂) * B̄') * (𝕀_big + T(0.5) * B))
end

function GeometricOptimizers._copyto!(
    Λ₁::GeometricOptimizers.GlobalSection{T, <:GrassmannManifold{T}},
    Λ₂::GeometricOptimizers.GlobalSection{T, <:GrassmannManifold{T}}
) where T
    copyto!(Λ₁.Y, Λ₂.Y)
    copyto!(Λ₁.λ, Λ₂.λ)
    Λ₁
end