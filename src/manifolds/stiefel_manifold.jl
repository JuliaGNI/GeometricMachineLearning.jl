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

Base.copy(A::StiefelManifold) = StiefelManifold(copy(A.A))
Base.similar(A::StiefelManifold) = StiefelManifold(similar(A.A))

function Base.zero(Y::StiefelManifold{T}) where T
    N, n = size(Y)
    backend = KernelAbstractions.get_backend(Y.A)
    zeros(backend, StiefelLieAlgHorMatrix{T}, N, n)
end

# Bridge GML's StiefelManifold into GO's manifold optimization infrastructure.
# GO's methods dispatch on MT<:GO.Manifold{T}, but GML.StiefelManifold<:GML.Manifold{T}
# (a different type hierarchy), so we extend GO's functions explicitly for GML's type.

function GeometricOptimizers.global_rep(
    λY::GeometricOptimizers.GlobalSection{T, <:StiefelManifold{T}},
    Δ::AbstractMatrix{T}
) where T
    N, n = size(λY.Y)
    StiefelLieAlgHorMatrix(
        SkewSymMatrix(λY.Y.A' * Δ),
        λY.λ' * Δ,
        N, n
    )
end

function GeometricOptimizers.update_section!(
    Λᵗ::GeometricOptimizers.GlobalSection{T, <:StiefelManifold{T}},
    Λ⁽ᵗ⁻¹⁾::GeometricOptimizers.GlobalSection{T, <:StiefelManifold{T}},
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

function GeometricOptimizers.cayley(B::StiefelLieAlgHorMatrix{T}) where T
    E = StiefelProjection(B)
    𝕀_small = one(B.A)
    𝕆 = zero(𝕀_small)
    𝕀_small2 = hcat(vcat(𝕀_small, 𝕆), vcat(𝕆, 𝕀_small))
    𝕀_big = one(B)
    A_mat = B.A * 𝕀_small
    B̂ = hcat(vcat(T(0.5) * A_mat, B.B), E)
    B̄ = hcat(vcat(𝕀_small, T(0.5) * A_mat), vcat(zero(B.B'), -B.B'))'
    StiefelManifold((𝕀_big + T(0.5) * B̂ * inv(𝕀_small2 - T(0.5) * B̄' * B̂) * B̄') * (𝕀_big + T(0.5) * B))
end

function GeometricOptimizers._copyto!(
    Λ₁::GeometricOptimizers.GlobalSection{T, <:StiefelManifold{T}},
    Λ₂::GeometricOptimizers.GlobalSection{T, <:StiefelManifold{T}}
) where T
    copyto!(Λ₁.Y, Λ₂.Y)
    copyto!(Λ₁.λ, Λ₂.λ)
    Λ₁
end

# GO arithmetic bridges for GML's SkewSymMatrix (GO dispatches by module, not structure).
GeometricOptimizers._add!(a::SkewSymMatrix{T}, b::SkewSymMatrix{T}) where T = (a.S .+= b.S; a)
GeometricOptimizers._add!(a::SkewSymMatrix{T}, b::T) where T = (a.S .+= b; a)
GeometricOptimizers._rac!(B::SkewSymMatrix, A::SkewSymMatrix) = (B.S .= sqrt.(A.S); B)
GeometricOptimizers._square!(B::SkewSymMatrix, A::SkewSymMatrix) = (B.S .= A.S .^ 2; B)
GeometricOptimizers._div!(C::SkewSymMatrix, A::SkewSymMatrix, B::SkewSymMatrix) = (C.S .= A.S ./ B.S; C)

# GO arithmetic bridges for GML's StiefelLieAlgHorMatrix.
function GeometricOptimizers._add!(A::StiefelLieAlgHorMatrix{T}, B::StiefelLieAlgHorMatrix{T}) where T
    GeometricOptimizers._add!(A.A, B.A); A.B .+= B.B; A
end
function GeometricOptimizers._add!(A::StiefelLieAlgHorMatrix{T}, b::T) where T
    GeometricOptimizers._add!(A.A, b); A.B .+= b; A
end
function GeometricOptimizers._rac!(B::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix)
    GeometricOptimizers._rac!(B.A, A.A); B.B .= sqrt.(A.B); B
end
function GeometricOptimizers._square!(B::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix)
    GeometricOptimizers._square!(B.A, A.A); B.B .= A.B .^ 2; B
end
function GeometricOptimizers._div!(C::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix, B::StiefelLieAlgHorMatrix)
    GeometricOptimizers._div!(C.A, A.A, B.A); C.B .= A.B ./ B.B; C
end