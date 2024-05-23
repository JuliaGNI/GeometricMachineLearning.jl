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
The geodesic map for the manifolds. It takes as input an element ``x`` of ``\mathcal{M}`` and an element of ``T_x\mathcal{M}`` and returns ``\mathtt{geodesic}(x, v_x) = \exp(v_x).`` For example: 

```julia 
Y = rand(StiefelManifold{Float64}, N, n)
Δ = rgrad(Y, rand(N, n))
geodesic(Y, Δ)
```

See the docstring for ``rgrad`` for details on this function.
"""
function geodesic(Y::Manifold{T}, Δ::AbstractMatrix{T}) where T
    λY = GlobalSection(Y)
    B = global_rep(λY, Δ)
    expB = geodesic(B)
    apply_section(λY, expB)
end

function geodesic(B::StiefelLieAlgHorMatrix{T}) where T
    E = StiefelProjection(B)
    unit = one(B.A)
    A_mat = B.A * unit
    exponent = hcat(vcat(T(.5) * A_mat, T(.25) * B.A * A_mat - B.B' * B.B), vcat(unit, T(.5) * A_mat))
    StiefelManifold(
        E + hcat(vcat(T(.5) * A_mat, B.B), E) * 𝔄(exponent) * vcat(unit, T(.5) * A_mat)
    )
end

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