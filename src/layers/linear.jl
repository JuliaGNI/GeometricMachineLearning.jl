import Lux

#activation layer
struct Linear{change_q, F1} <: Lux.AbstractExplicitLayer
        init_weight::F1
        dim::Int
end

#change random number generator to make SymmetricMatrix
function Linear(dim::Int; change_q::Bool=true, init_weight=Lux.glorot_uniform)
        iseven(dim) || error("Dimension must be even!")
        return Linear{change_q, typeof(init_weight)}(init_weight, dim)
end

function Lux.initialparameters(rng::AbstractRNG, d::Linear)
    return (weight=SymmetricMatrix(d.init_weight(rng, d.dim÷2, d.dim÷2)),)
end

Lux.initialstates(rng::AbstractRNG, d::Linear) = NamedTuple()

#somewhat tricky because of the parametrization of a symmetric matrix
function Lux.parameterlength(d::Linear{change_q}) where {change_q}
        return (d.dim÷2+1)*d.dim÷4
end

@inline function(d::Linear{true})(x::AbstractVecOrMat, ps, st::NamedTuple)
        return vcat(x[1:(d.dim÷2)] + ps.weight*x[(d.dim÷2+1):d.dim], x[(d.dim÷2+1):d.dim]),st
end

@inline function(d::Linear{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
        return vcat(x[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps.weight*x[1:(d.dim÷2)]),st
end
