using Lux
using Random
using NNlib



#activation layer
struct Gradient{full_grad, F1, F2, F3, F4} <: Lux.AbstractExplicitLayer
        activation::F1
        dim::Int
        dim2::Int
        init_weight::F2
        init_bias::F3
        init_scale::F4
        change_q::Bool
end


#check: input is even; make dim2 an optional argument for full_grad=false
function Gradient(dim::Int, dim2::Int, activation=identity; init_weight=Lux.glorot_uniform,
                init_bias=Lux.zeros32, init_scale=Lux.glorot_uniform, full_grad::Bool=true,
                change_q::Bool=true, allow_fast_activation::Bool=true)
        activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
        dtype = (full_grad, typeof(activation), typeof(init_weight), typeof(init_bias), 
                 typeof(init_scale))
        return Gradient{dtype...}(activation, dim, dim2, init_weight, init_bias, init_scale, change_q)
end


function Lux.initialparameters(rng::AbstractRNG, d::Gradient{full_grad}) where {full_grad}
        if full_grad
                return (weight=d.init_weight(rng, d.dim2, d.dim),
                        bias=d.init_bias(rng, d.dim2, 1),
                        scale=d.init_scale(rng,d.dim2,1))
        else
                return (scale=d.init_scale(rng, d.dim, 1),)
        end
end

Lux.initialstates(rng::AbstractRNG, d::Gradient) = NamedTuple()

function Lux.parameterlength(d::Gradient{full_grad}) where {full_grad}
        return full_grad ? d.dim2 * (d.dim + 2) : d.dim
end
Lux.statelength(d::Gradient) = 0

@inline function (d::Gradient{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
        if d.change_q
                return vcat(x[1:d.dim] + ps.scale.*d.activation.(x[d.dim+1:2*d.dim]),
                                                                 x[d.dim+1:2*d.dim]), st
        else
                return vcat(x[1:d.dim], x[d.dim+1:2*d.dim] + ps.scale.*
                        d.activation.(x[1:d.dim])),st

        end
end
#implement another routine for AbstractArray?

@inline function (d::Gradient{true})(x::AbstractVecOrMat, ps, st::NamedTuple)
        if d.change_q
                return vcat(x[1:d.dim] + ps.weight' * (ps.scale .* 
                        d.activation.(ps.weight * x[d.dim+1:2*d.dim] .+ vec(ps.bias))), 
                        x[d.dim+1:2*d.dim]), st
            else 
                return vcat(x[1:d.dim], x[d.dim+1:2*d.dim] + ps.weight' * 
                            (ps.scale .* d.activation(ps.weight*x[1:d.dim] .+ vec(ps.bias)))), st
            end
end

#implement another routine for AbstractMatrix?

###short test
dummy_model = Gradient(2,5,tanh)
ps,st = Lux.setup(Random.default_rng(), dummy_model)
dummy_model(ones(4),ps,st)[1]




