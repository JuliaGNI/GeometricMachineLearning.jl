#activation layer
struct Grad{full_grad, F1, F2, F3, F4} <: Lux.AbstractExplicitLayer
        activation::F1
        dim::Int
        dim2::Int
        init_weight::F2
        init_bias::F3
        init_scale::F4
end

#need WrappedFunction, Scale and Parallel
#
#check: input is even; somehow split input
function Grad(dim::Int, dim2::Int, activation=identity; init_weight=Lux.glorot_uniform,
                init_bias=Lux.zeros32, init_scale=Lux.glorot_uniform, full_grad::Bool=true,
                allow_fast_activation::Bool=true)
        activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
        dtype = (full_grad, typeof(activation), typeof(init_weight), typeof(init_bias), 
                 typeof(init_scale))
        return Grad{dtype...}(activation, dim, dim2, init_weight, init_bias, init_scale)
end


function Lux.initialparameters(rng::AbstractRNG, d::Grad{full_grad}) where {full_grad}
        if full_grad
                return (weight=d.init_weight(rng, d.dim2, d.dim),
                        bias=d.init_bias(rng, d.dim2, 1),
                        scale=d.init_scale(rng,d.dim2,1))
        else
                return (scale=d.init_scale(rng, d.dim, 1),)
        end
end

Lux.initialstates(rng::AbstractRNG, d::Grad) = NamedTuple()

function Lux.parameterlength(d::Grad{full_grad}) where {full_grad}
        return full_grad ? d.dim2 * (d.dim + 2) : d.dim
end
Lux.statelength(d::Grad) = 0

@inline function (d::Grad{false})(x::AbstractVector, ps, st::NamedTuple)
        return vcat(x[1:d.dim] + ps.scale.*d.activation.(x[d.dim+1:2*d.dim],
                                                         x[d.dim+1:2*d.dim]), st)
end

#implement another routine for AbstractArray?

@inline function (d::Grad{true})(x::AbstractVector, ps, st::NamedTuple)
    return vcat(x[1:d.dim] + ps.weight' * (ps.scale .* 
                d.activation.(ps.weight * x[d.dim+1:2*d.dim] .+ vec(ps.bias))), 
                x[d.dim+1:2*d.dim]), st
end

#implement another routine for AbstractMatrix?

###
dummy_model = Grad(2,5,tanh)
ps,st = Lux.setup(Random.default_rng(), dummy_model)


