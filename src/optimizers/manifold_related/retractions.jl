geodesic(A::AbstractVecOrMat) = A
cayley(A::AbstractVecOrMat) = A

geodesic(B::NamedTuple) = apply_toNT(geodesic, B)

@doc raw"""
    geodesic(Y::Manifold, Δ)

Take as input an element of a manifold `Y` and a tangent vector in `Δ` in the corresponding tangent space and compute the geodesic (exponential map).

In different notation: take as input an element ``x`` of ``\mathcal{M}`` and an element of ``T_x\mathcal{M}`` and return ``\mathtt{geodesic}(x, v_x) = \exp(v_x).``


# Examples

```jldoctest
using GeometricMachineLearning

Y = StiefelManifold([1. 0. 0.;]' |> Matrix)
Δ = [0. .5 0.;]' |> Matrix
Y₂ = geodesic(Y, Δ)

Y₂' * Y₂ ≈ [1.;]

# output

true
```

# Implementation

Internally this `geodesic` method calls [`geodesic(::StiefelLieAlgHorMatrix)`](@ref).
"""
function geodesic(Y::Manifold{T}, Δ::AbstractMatrix{T}) where T
    λY = GlobalSection(Y)
    B = global_rep(λY, Δ)
    E = StiefelProjection(B)
    expB = geodesic(B)
    λY * typeof(Y)(expB * E)
end

@doc raw"""
    geodesic(B̄::StiefelLieAlgHorMatrix)

Compute the geodesic of an element in [`StiefelLieAlgHorMatrix`](@ref).

# Implementation

Internally this is using:

```math
\mathbb{I} + B'\mathfrak{A}(B', B'')B'',
```

with 

```math
\bar{B} = \begin{bmatrix}
    A & -B^T \\ 
    B & \mathbb{O}
\end{bmatrix} = \begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix} =: B'(B'')^T.
```

This is using a computationally efficient version of the matrix exponential ``\mathfrak{A}``. 

See [`GeometricMachineLearning.𝔄`](@ref).
"""
function geodesic(B::StiefelLieAlgHorMatrix)
    T = eltype(B)
    E = StiefelProjection(B)
    unit = one(B.A)
    A_mat = B.A * unit
    B̂ = hcat(vcat(T(.5) * A_mat, B.B), E)
    B̄ = hcat(vcat(unit, T(.5) * A_mat), vcat(zero(B.B'), -B.B'))'
    StiefelManifold(one(B) + B̂ * 𝔄(B̂, B̄) * B̄')
end

@doc raw"""
    geodesic(B̄::GrassmannLieAlgHorMatrix)

Compute the geodesic of an element in [`GrassmannLieAlgHorMatrix`](@ref).

This is equivalent to the method of [`geodesic`](@ref) for [StiefelLieAlgHorMatrix](@ref).

See [`geodesic(::StiefelLieAlgHorMatrix)`](@ref).
"""
function geodesic(B::GrassmannLieAlgHorMatrix)
    T = eltype(B)
    E = StiefelProjection(B)
    backend = KernelAbstractions.get_backend(B)
    zero_mat = KernelAbstractions.zeros(backend, T, B.n, B.n)
    B̂ = hcat(vcat(zero_mat, B.B), E)
    B̄ = hcat(vcat(one(zero_mat), zero_mat), vcat(zero(B.B'), -B.B'))'
    GrassmannManifold(one(B) + B̂ * 𝔄(B̂, B̄) * B̄')
end

cayley(B::NamedTuple) = apply_toNT(cayley, B)

@doc raw"""
    cayley(Y::Manifold, Δ)

Take as input an element of a manifold `Y` and a tangent vector in `Δ` in the corresponding tangent space and compute the Cayley retraction.

In different notation: take as input an element ``x`` of ``\mathcal{M}`` and an element of ``T_x\mathcal{M}`` and return ``\mathrm{Cayley}(v_x).`` 

# Examples

```jldoctest
using GeometricMachineLearning

Y = StiefelManifold([1. 0. 0.;]' |> Matrix)
Δ = [0. .5 0.;]' |> Matrix
Y₂ = cayley(Y, Δ)

Y₂' * Y₂ ≈ [1.;]

# output

true
```

See the example in [`geodesic(::Manifold{T}, ::AbstractMatrix{T}) where T`].
"""
function cayley(Y::Manifold{T}, Δ::AbstractMatrix{T}) where T
    λY = GlobalSection(Y)
    B = global_rep(λY, Δ)
    E = StiefelProjection(B)
    cayleyB = cayley(B)
    λY * typeof(Y)(cayleyB * E)
end

@doc raw"""
    cayley(B̄::StiefelLieAlgHorMatrix)

Compute the Cayley retraction of `B`.

# Implementation

Internally this is using 

```math
\mathrm{Cayley}(\bar{B}) = \mathbb{I} + \frac{1}{2} B' (\mathbb{I}_{2n} - \frac{1}{2} (B'')^T B')^{-1} (B'')^T (\mathbb{I} + \frac{1}{2} B),
```
with
```math
\bar{B} = \begin{bmatrix}
    A & -B^T \\ 
    B & \mathbb{O}
\end{bmatrix} = \begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix} =: B'(B'')^T,
```
i.e. ``\bar{B}`` is expressed as a product of two ``N\times{}2n`` matrices.
"""
function cayley(B::StiefelLieAlgHorMatrix)
    T = eltype(B)
    E = StiefelProjection(B)
    𝕀_small = one(B.A)
    𝕆 = zero(𝕀_small)
    𝕀_small2 = hcat(vcat(𝕀_small, 𝕆), vcat(𝕆, 𝕀_small))
    𝕀_big = one(B)
    A_mat = B.A * 𝕀_small
    B̂ = hcat(vcat(T(.5) * A_mat, B.B), E)
    B̄ = hcat(vcat(𝕀_small, T(.5) * A_mat), vcat(zero(B.B'), -B.B'))'

    StiefelManifold((𝕀_big + T(.5) * B̂ * inv(𝕀_small2 - T(.5) * B̄' * B̂) * B̄') * (𝕀_big + T(.5) * B))
end

@doc raw"""
    cayley(B̄::GrassmannLieAlgHorMatrix)

Compute the Cayley retraction of `B`.

This is equivalent to the method of [`cayley`](@ref) for [StiefelLieAlgHorMatrix](@ref).

See [`cayley(::StiefelLieAlgHorMatrix)`](@ref).
"""
function cayley(B::GrassmannLieAlgHorMatrix)
    T = eltype(B)
    E = StiefelProjection(B)
    backend = KernelAbstractions.get_backend(B)
    𝕆 = KernelAbstractions.zeros(backend, T, B.n, B.n)
    𝕀_small = one(𝕆)
    𝕀_small2 = hcat(vcat(𝕀_small, 𝕆), vcat(𝕆, 𝕀_small))
    𝕀_big = one(B)
    B̂ = hcat(vcat(𝕆, B.B), E)
    B̄ = hcat(vcat(𝕀_small, 𝕆), vcat(zero(B.B'), -B.B'))'

    GrassmannManifold((𝕀_big + T(.5) * B̂ * inv(𝕀_small2 - T(.5) * B̄' * B̂) * B̄') * (𝕀_big + T(.5) * B))
end