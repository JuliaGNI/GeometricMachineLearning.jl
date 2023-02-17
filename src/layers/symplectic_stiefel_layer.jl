
import SparseArrays
import Manifolds

"""
n is input dimension (small);
N is output dimension (big)
"""
struct SymplecticStiefelLayer{inverse, F1, MT <: Manifolds.SymplecticStiefel} <:
       ManifoldLayer
    dim_n::Int
    dim_N::Int
    manifold::MT
    initial_weight::F1
    sympl_in::SparseArrays.SparseMatrixCSC
    sympl_out::SparseArrays.SparseMatrixCSC
end

#maybe implement another random number generator
function SymplecticStiefelLayer(dim_n::Int, dim_N::Int; inverse::Bool = false)
    iseven(dim_n) && iseven(dim_N) || error("Dimension must be even.")
    dim_n ≤ dim_N || error("Output dimension must be bigger than input dimension.")
    M = Manifolds.SymplecticStiefel(dim_N, dim_n)
    init_weight = () -> rand(M)
    SymplecticStiefelLayer{inverse, typeof(init_weight), typeof(M)}(dim_n, dim_N, M,
                                                                    init_weight,
                                                                    SymplecticMatrix(dim_n ÷
                                                                                     2),
                                                                    SymplecticMatrix(dim_N ÷
                                                                                     2))
end

function Lux.initialparameters(RNG::AbstractRNG, d::SymplecticStiefelLayer)
    (weight = d.initial_weight(),)
end

function Lux.initialstates(RNG::AbstractRNG, d::SymplecticStiefelLayer)
    (IsManifold = true, Manifold = d.manifold)
end

function Lux.parameterlength(d::SymplecticStiefelLayer)
    (4 * d.dim_N - 2 * d.dim_n + 1) * d.dim_n
end

Lux.statelength(d::SymplecticStiefelLayer) = 2

@inline function (d::SymplecticStiefelLayer{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
    ps.weight * x, st
end

@inline function (d::SymplecticStiefelLayer{true})(x::AbstractVecOrMat, ps, st::NamedTuple)
    -d.sympl_in * ps.weight' * d.sympl_out * x, st
end

function update_layer!(l::SymplecticStiefelLayer, x, dx, η)
    Manifolds.retract_caley!(l.manifold, x.weight, copy(x.weight), -η * dx)
end
