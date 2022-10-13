using Lux
using Random
using NNlib
using LinearAlgebra

include("../src/arrays/symmetric.jl")


#activation layer
struct Linear{change_q, F1} <: Lux.AbstractExplicitLayer
        init_weight::F1
        dim::Int
end

#change random number generator to make SymmetricMatrix
function Linear(dim::Int; change_q::Bool=true, init_weight=Lux.glorot_uniform)
        return Linear{change_q, typeof(init_weight)}(init_weight, dim)
end

function Lux.initialparameters(rng::AbstractRNG, d::Linear)
    return (weight=SymmetricMatrix(d.init_weight(rng, d.dim, d.dim)),)
end

Lux.initialstates(rng::AbstractRNG, d::Linear) = NamedTuple()

#somewhat tricky because of the parametrization of a symmetric matrix
function Lux.parameterlength(d::Linear{change_q}) where {change_q}
        return (Linear.dim+1)*Linear.dim÷2
end

@inline function(d::Linear{true})(x::AbstractVecOrMat, ps, st::NamedTuple)
        return vcat(x[1:d.dim] + ps.weight*x[d.dim+1:2*d.dim], x[d.dim+1:2*d.dim]),st
end

@inline function(d::Linear{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
        return vcat(x[1:d.dim], x[d.dim+1:2*d.dim] + ps.weight*x[1:d.dim]),st
end

###short test
dummy_model = Linear(2,change_q=false)
ps,st = Lux.setup(Random.default_rng(), dummy_model)
print(dummy_model(ones(4),ps,st)[1])




