struct PostProcessing{F1, F2, F3} <: Lux.AbstractExplicitLayer
    activation::F1
    dim::Int
    init_weight::F2
    init_bias::F3
end

function PostProcessing(dim::Int, activation=tanh; init_weight=Lux.glorot_uniform,
               init_bias=Lux.zeros32, allow_fast_activation::Bool=true)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation

    dtype = (typeof(activation), typeof(init_weight), typeof(init_bias))
    return PostProcessing{dtype...}(activation, dim, init_weight, init_bias)
end

function Lux.initialparameters(rng::Random.AbstractRNG, d::PostProcessing)
    return (weight1=d.init_weight(rng, d.dim, d.dim), weight2=d.init_weight(rng, d.dim, d.dim), bias1=d.init_bias(rng, d.dim), bias2=d.init_bias(rng, d.dim))
end

function Lux.parameterlength(d::PostProcessing) 
    return 2*d.dim * (d.dim + 1) 
end
Lux.statelength(d::ResNet) = 0

function Lux.apply(d::PostProcessing, x::AbstractVecOrMat, ps::NamedTuple, st::NamedTuple)
    return x + ps.weight2*d.activation(ps.weight1*x + ps.bias1) + ps.bias2, st
end

function Lux.apply(d::PostProcessing, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple) where T 
	return x + mat_tensor_mul(ps.weight2, d.activation(mat_tensor_mul(ps.weight1, x) .+ ps.bias1)) .+ ps.bias2, st
end
