using Lux
using Random

import Manifolds

struct SymplecticStiefelLayer{inverse, F1, MT <: Manifolds.SymplecticStiefel} <: Lux.AbstractExplicitLayer
    dim_in::Int
    dim_out::Int
    manifold::MT
    initial_weight::F1
end

#maybe implement another random number generator
function SymplecticStiefelLayer(dim_in::Int, dim_out::Int; inverse::Bool = false)
    M = Manifolds.SymplecticStiefel(dim_out, dim_in)
    init_weight = () -> rand(M, 1)[1]
    SymplecticStiefelLayer{inverse, typeof(init_weight), typeof(M)}(dim_in, dim_out, M, init_weight)
end

function Lux.initialparameters(RNG::AbstractRNG, d::SymplecticStiefelLayer)
    (weight = d.initial_weight(),)
end

function Lux.initialstates(RNG::AbstractRNG, d::SymplecticStiefelLayer)
    (Manifold = true, ManifoldType = Manifolds.SymplecticStiefel)
end

function Lux.parameterlength(d::SymplecticStiefelLayer)
    (4 * d.dim_out - 2 * d.dim_in + 1) * d.dim_in
end

Lux.statelength(d::SymplecticStiefelLayer) = 0

@inline function (d::SymplecticStiefelLayer{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
    ps.weight * x, st
end

#maybe pick the symplectic inverse here
@inline function (d::SymplecticStiefelLayer{true})(x::AbstractVecOrMat, ps, st::NamedTuple)
    ps.weight' * x, st
end
