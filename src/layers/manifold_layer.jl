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
    iseven(dim_in) && iseven(dim_out) || error("Dimension must be even.")
    dim_in ≤ dim_out || error("Output dimension must be bigger than input dimension.")
    M = Manifolds.SymplecticStiefel(dim_out, dim_in)
    init_weight = () -> rand(M)
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

symplectic_flip(x::AbstractVector,n::Int) = iseven(n) ? vcat(x[(n÷2+1):n],-x[1:(n÷2)]) : error("Dimension must be even.")


@inline function (d::SymplecticStiefelLayer{true})(x::AbstractVecOrMat, ps, st::NamedTuple)
    -symplectic_flip(ps.weight' * symplectic_flip(x,d.dim_out),d.dim_in), st
end


##tests

using Zygote
using Random


model1 = SymplecticStiefelLayer(10,20;inverse=true)
ps, st = Lux.setup(Random.default_rng(),model1)
gradient(p -> sum(Lux.apply(model1, rand(20), p, st)[1]), ps)[1]


model2 = Chain(Gradient(200,1000),Gradient(200,500),SymplecticStiefelLayer(20,200;inverse=true),Gradient(20,50))
ps, st = Lux.setup(Random.default_rng(),model2)
gradient(p -> sum(Lux.apply(model2, rand(200), p, st)[1]), ps)[1]
