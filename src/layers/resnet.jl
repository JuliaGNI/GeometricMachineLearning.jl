@doc raw"""
    ResNetLayer(dim)

Make an instance of the resnet layer.

The `ResNetLayer` is a simple feedforward neural network to which we add the input after applying it, i.e. it realizes ``x \mapsto x + \sigma(Ax + b)``.

# Arguments

The ResNet layer takes the following arguments:
1. `dim::Integer`: the system dimension.
2. `activation = identity`: The activation function.

The following is a keyword argument:
- `use_bias::Bool = true`: This determines whether a bias ``b`` is used.
"""
struct ResNetLayer{M, N, use_bias, F1} <: AbstractExplicitLayer{M, N}
    activation::F1
end

function ResNetLayer(dim::Integer, activation=identity; use_bias::Bool=true)
    return ResNetLayer{dim, dim, use_bias, typeof(activation)}(activation)
end

function initialparameters(rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer, ::ResNetLayer{M, M, use_bias}, backend::KernelAbstractions.Backend, ::Type{T}; init_bias = ZeroInitializer()) where {M, use_bias, T}
    if use_bias
        weight = KernelAbstractions.allocate(backend, T, M, M)
        bias = KernelAbstractions.allocate(backend, T, M)
        init_weight(rng, weight)
        init_bias(rng, bias)
        return (weight=weight,
                bias=bias)
    else
        weight = KernelAbstractions.allocate(backend, T, M, M)
        init_weight(rng, weight)
        return (weight=weight,)
    end
end

function parameterlength(::ResNetLayer{M, M, use_bias}) where {M, use_bias}
    return use_bias ? M * (M + 1) : M * M
end

@inline function (d::ResNetLayer{M, M, true})(x::AbstractVecOrMat, ps::NamedTuple) where {M}
    return x + d.activation.(ps.weight * x .+ ps.bias)
end

@inline function (d::ResNetLayer{M, M, false})(x::AbstractVecOrMat, ps::NamedTuple) where {M}
    return x + d.activation.(ps.weight * x)
end

@inline function (d::ResNetLayer{M, M, false})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, T}
    return x + d.activation.(mat_tensor_mul(ps.weight, x))
end

@inline function (d::ResNetLayer{M, M, true})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, T}
    return x + d.activation.(mat_tensor_mul(ps.weight, x) .+ ps.bias)
end

@inline function (d::Dense{M, N, true})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T}
	return d.σ.(mat_tensor_mul(ps.W, x) .+ ps.b)
end

@inline function (d::Dense{M, N, false})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T}
	return d.σ.(mat_tensor_mul(ps.W, x))
end

function (d::Union{Dense{M, N}, ResNetLayer{M, N}})(z::QPT, ps::NamedTuple) where {M, N}
    @assert iseven(M) == iseven(N) == true
    @assert size(z.q, 1) * 2 == M
    N2 = N ÷ 2
    output = d(vcat(z.q, z.p), ps)
    assign_q_and_p(output, N2)
end