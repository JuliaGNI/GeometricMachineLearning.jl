import Lux

#linear layer with bias or not
struct Linear{bias, change_q, F1, F2} <: Lux.AbstractExplicitLayer
        init_weight::F1
        init_bias::F2
        dim::Int
end

#change random number generator to make SymmetricMatrix
function Linear(dim::Int; change_q::Bool=true, bias::Bool = false, init_weight=Lux.glorot_uniform, init_bias=Lux.zeros32)
        iseven(dim) || error("Dimension must be even!")
        return Linear{bias,change_q,typeof(init_weight),typeof(init_bias)}(init_weight, init_bias, dim)
end

function Lux.initialparameters(rng::AbstractRNG, d::Linear{bias}) where {bias}
        if bias
                return (weight=SymmetricMatrix(d.init_weight(rng, d.dim÷2, d.dim÷2)), bias=d.init_bias(rng,d.dim))
        else
                return (weight=SymmetricMatrix(d.init_weight(rng, d.dim÷2, d.dim÷2)),)
        end
end

Lux.initialstates(rng::AbstractRNG, d::Linear) = NamedTuple()

#somewhat tricky because of the parametrization of a symmetric matrix
function Lux.parameterlength(d::Linear{bias}) where {bias}
        return bias ? (d.dim÷2+1)*d.dim÷4 + d.dim : (d.dim÷2+1)*d.dim÷4
end

@inline function(d::Linear{false,true})(x::AbstractVecOrMat, ps, st::NamedTuple)
        return vcat(x[1:(d.dim÷2)] + ps.weight*x[(d.dim÷2+1):d.dim], x[(d.dim÷2+1):d.dim]),st
end

@inline function(d::Linear{false,false})(x::AbstractVecOrMat, ps, st::NamedTuple)
        return vcat(x[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps.weight*x[1:(d.dim÷2)]),st
end


@inline function(d::Linear{true,true})(x::AbstractVecOrMat, ps, st::NamedTuple)
        return vcat(x[1:(d.dim÷2)] + ps.weight*x[(d.dim÷2+1):d.dim]+ps.bias[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim]+ ps.bias[(d.dim÷2+1):d.dim]),st
end

@inline function(d::Linear{true,false})(x::AbstractVecOrMat, ps, st::NamedTuple)
        return vcat(x[1:(d.dim÷2)]+ps.bias[1:(d.dim÷2)], x[(d.dim÷2+1):d.dim] + ps.weight*x[1:(d.dim÷2)] + ps.bias[(d.dim÷2+1):d.dim]),st
end
