@doc raw"""
`LayerWithManifold` is a subtype of `AbstractExplicitLayer` that contains manifolds as weights.
"""
abstract type LayerWithManifold{M, N, retraction} <: AbstractExplicitLayer{M, N}  end

@doc raw"""
`LayerWithOptionalManifold` is a subtype of `AbstractExplicitLayer` that can contain manifolds as weights.
"""
abstract type LayerWithOptionalManifold{M, N, Stiefel, retraction} <: AbstractExplicitLayer{M, N} end

#fallback function -> maybe put into another file!
function retraction(::AbstractExplicitLayer, gx::NamedTuple)
    gx
end

function retraction(::LayerWithManifold{M, N, Geodesic}, B::NamedTuple) where {M,N}
    geodesic(B)
end
  
function retraction(::AbstractExplicitCell, gx::NamedTuple)
    gx
end

function retraction(::LayerWithManifold{M, N, Cayley}, B::NamedTuple) where {M,N}
    cayley(B)
end

function retraction(::LayerWithOptionalManifold{M, N, true, Geodesic}, B::NamedTuple) where {M,N}
    geodesic(B)
end

function retraction(::LayerWithOptionalManifold{M, N, true, Cayley}, B::NamedTuple) where {M,N}
    cayley(B)
end

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
    expB = geodesic(B)
    apply_section(λY, expB)
end

@doc raw"""
    geodesic(B::StiefelLieAlgHorMatrix)

Compute the geodesic of `B*E` where `E` is the distinct element of the StiefelManifold.

# Implementation

This is using a computationally efficient version of the matrix exponential. See [`GeometricMachineLearning.𝔄`](@ref).
"""
function geodesic(B::StiefelLieAlgHorMatrix{T}) where T
    E = StiefelProjection(B)
    unit = one(B.A)
    A_mat = B.A * unit
    exponent = hcat(vcat(T(.5) * A_mat, T(.25) * B.A * A_mat - B.B' * B.B), vcat(unit, T(.5) * A_mat))
    StiefelManifold(
        E + hcat(vcat(T(.5) * A_mat, B.B), E) * 𝔄(exponent) * vcat(unit, T(.5) * A_mat)
    )
end

@doc raw"""
    geodesic(B::GrassmannLieAlgHorMatrix)

Compute the geodesic of `B*E` where `E` is the distinct element of the StiefelManifold.

See [`geodesic(::StiefelLieAlgHorMatrix)`](@ref).
"""
function geodesic(B::GrassmannLieAlgHorMatrix{T}) where T
    N, n = B.N, B.n
    E = typeof(B.B)(StiefelProjection(N, n, T))
    # expression from which matrix exponential and inverse have to be computed
    unit = typeof(B.B)(I(n))
    exponent = hcat(vcat(zeros(T, n, n), - B.B' * B.B), vcat(unit, zeros(T, n, n)))
    GrassmannManifold(
        E + (hcat(vcat(zeros(T, n, n), B.B), E) * 𝔄(exponent))[1:N, 1:n]
    )
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
    cayleyB = cayley(B)
    apply_section(λY, cayleyB)
end

@doc raw"""
    cayley(B::StiefelLieAlgHorMatrix)

Compute the Cayley retraction of `B` and multiply it with `E` (the distinct element of the Stiefel manifold).
"""
function cayley(B::StiefelLieAlgHorMatrix{T}) where T
    E = StiefelProjection(B)
    unit = one(B.A)
    A_mat = B.A * one(B.A)
    A_mat2 = B.A * B.A 
    BB = B.B' * B.B

    exponent = hcat(vcat(unit - T(.25) * A_mat, T(.5) * BB - T(.125) * A_mat2), vcat(-T(.5) * unit, unit - T(.25) * A_mat))
    StiefelManifold(
        E + 
        T(.5) * hcat(vcat(T(.5) * A_mat, B.B), vcat(unit, zero(B.B)))*
        (
            vcat(unit, T(0.5) * A_mat) + exponent \ (vcat(unit, T(0.5) * A_mat) + vcat(T(0.5) * A_mat, T(0.25) * A_mat2 - T(0.5) * BB))
            )
    )
end

@doc raw"""
    cayley(B::GrassmannLieAlgHorMatrix)

Compute the Cayley retraction of `B` and multiply it with `E` (the distinct element of the Stiefel manifold).

See [`cayley(::StiefelLieAlgHorMatrix)`](@ref).
"""
function cayley(B::GrassmannLieAlgHorMatrix{T}) where T
    error("Missing implementation!")
end