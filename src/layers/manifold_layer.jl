using Lux
using Random
using LinearAlgebra

import SparseArrays
import Manifolds

"""
n is input dimension (small);
N is output dimension (big)
"""
struct SymplecticStiefelLayer{inverse, F1, MT <: Manifolds.SymplecticStiefel} <: Lux.AbstractExplicitLayer
    dim_n::Int
    dim_N::Int
    manifold::MT
    initial_weight::F1
    sympl_in::SparseArrays.SparseMatrixCSC
    sympl_out::SparseArrays.SparseMatrixCSC
end

make_sympl_mat(n) = hcat(vcat(zeros(n÷2,n÷2),-I(n÷2)),vcat(I(n÷2),zeros(n÷2,n÷2)))

#maybe implement another random number generator
function SymplecticStiefelLayer(dim_in::Int, dim_out::Int; inverse::Bool = false)
    iseven(dim_in) && iseven(dim_out) || error("Dimension must be even.")
    dim_in ≤ dim_out || error("Output dimension must be bigger than input dimension.")
    M = Manifolds.SymplecticStiefel(dim_out, dim_in)
    init_weight = () -> rand(M)
    SymplecticStiefelLayer{inverse, typeof(init_weight), typeof(M)}(dim_in, dim_out, M, init_weight,
    make_sympl_mat(dim_in), make_sympl_mat(dim_out))
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


@inline function (d::SymplecticStiefelLayer{true})(x::AbstractVecOrMat, ps, st::NamedTuple)
    -d.sympl_in * ps.weight' * d.sympl_out * x, st
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
