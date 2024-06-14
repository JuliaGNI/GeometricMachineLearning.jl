@doc raw"""
`LayerWithManifold` is a subtype of `AbstractExplicitLayer` that contains manifolds as weights.
"""
abstract type LayerWithManifold{M, N, retraction} <: AbstractExplicitLayer{M, N}  end

@doc raw"""
`LayerWithOptionalManifold` is a subtype of `AbstractExplicitLayer` that can contain manifolds as weights.
"""
abstract type LayerWithOptionalManifold{M, N, Stiefel, retraction} <: AbstractExplicitLayer{M, N} end

geodesic(A::AbstractVecOrMat) = A
cayley(A::AbstractVecOrMat) = A

geodesic(B::NamedTuple) = apply_toNT(geodesic, B)

@doc raw"""
    geodesic(Y::Manifold, Δ)

Take as input an element of a manifold `Y` and a tangent vector in `Δ` in the corresponding tangent space and compute the geodesic (exponential map).

In different notation: take as input an element ``x`` of ``\mathcal{M}`` and an element of ``T_x\mathcal{M}`` and return ``\mathtt{geodesic}(x, v_x) = \exp(v_x).`` For example: 

```julia 
Y = rand(StiefelManifold{Float64}, N, n)
Δ = rgrad(Y, rand(N, n))
geodesic(Y, Δ)
```

See the docstring for [`rgrad`](@ref) for details on this function.
"""
function geodesic(Y::Manifold{T}, Δ::AbstractMatrix{T}) where T
    λY = GlobalSection(Y)
    B = global_rep(λY, Δ)
    E = StiefelProjection(B)
    expB = geodesic(B)
    λY * typeof(Y)(expB * E)
end

@doc raw"""
    geodesic(B::StiefelLieAlgHorMatrix)

Compute the geodesic of an element in [`StiefelLieAlgHorMatrix`](@ref).

# Implementation

This is using a computationally efficient version of the matrix exponential. See [`GeometricMachineLearning.𝔄`](@ref).
"""
function geodesic(B::StiefelLieAlgHorMatrix{T}) where T
    E = StiefelProjection(B)
    unit = one(B.A)
    A_mat = B.A * unit
    B̂ = hcat(vcat(T(.5) * A_mat, B.B), E)
    B̄ = hcat(vcat(unit, T(.5) * A_mat), vcat(zero(B.B'), -B.B'))'
    StiefelManifold(one(B) + B̂ * 𝔄(B̂, B̄) * B̄')
end

@doc raw"""
    geodesic(B::GrassmannLieAlgHorMatrix)

Compute the geodesic of an element in [`GrassmannLieAlgHorMatrix`](@ref).

See [`geodesic(::StiefelLieAlgHorMatrix)`](@ref).
"""
function geodesic(B::GrassmannLieAlgHorMatrix{T}) where T
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

In different notation: take as input an element ``x`` of ``\mathcal{M}`` and an element of ``T_x\mathcal{M}`` and return ``\mathrm{Cayley}(v_x).`` For example: 

```julia 
Y = rand(StiefelManifold{Float64}, N, n)
Δ = rgrad(Y, rand(N, n))
cayley(Y, Δ)
```

See the docstring for [`rgrad`](@ref) for details on this function.
"""
function cayley(Y::Manifold{T}, Δ::AbstractMatrix{T}) where T
    λY = GlobalSection(Y)
    B = global_rep(λY, Δ)
    E = StiefelProjection(B)
    cayleyB = cayley(B)
    λY * typeof(Y)(cayleyB * E)
end

@doc raw"""
    cayley(B::StiefelLieAlgHorMatrix)

Compute the Cayley retraction of `B` and multiply it with `E` (the distinct element of the Stiefel manifold).
"""
function cayley(B::StiefelLieAlgHorMatrix{T}) where T
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
    cayley(B::GrassmannLieAlgHorMatrix)

Compute the Cayley retraction of `B` and multiply it with `E` (the distinct element of the Stiefel manifold).

See [`cayley(::StiefelLieAlgHorMatrix)`](@ref).
"""
function cayley(B::GrassmannLieAlgHorMatrix{T}) where T
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