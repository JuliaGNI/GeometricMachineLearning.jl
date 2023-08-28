struct ResNet{M, N, use_bias, F1} <: AbstractExplicitLayer{M, N}
    activation::F1
end

function ResNet(dim::IT, activation=identity; use_bias::Bool=true) where {IT<:Int}
    return ResNet{dim, dim, use_bias, typeof(activation)}(activation)
end

function initialparameters(backend::KernelAbstractions.Backend, T::Type, ::ResNet{M, M, use_bias}; rng::Random.AbstractRNG=Random.default_rng(), init_weight = GlorotUniform(), init_bias = ZeroInitializer()) where {M, use_bias}
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

function parameterlength(::ResNet{M, M, use_bias}) where {M, use_bias}
    return use_bias ? M * (M + 1) : M * M
end

@inline function (d::ResNet{M, M, true})(x::AbstractVecOrMat, ps::NamedTuple) where {M}
    return x + d.activation.(ps.weight * x .+ ps.bias)
end

@inline function (d::ResNet{M, M, false})(x::AbstractVecOrMat, ps::NamedTuple) where {M}
    return x + d.activation.(ps.weight * x)
end

@inline function (d::ResNet{M, M, false})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, T}
    return x + d.activation.(mat_tensor_mul(ps.weight, x))
end

@inline function (d::ResNet{M, M, true})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, T}
    return x + d.activation.(mat_tensor_mul(ps.weight, x) .+ ps.bias)
end

@inline function (d::Dense{M, N, true})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T}
	return d.σ.(mat_tensor_mul(ps.W, x) .+ ps.b)
end

@inline function (d::Dense{M, N, false})(x::AbstractArray{T, 3}, ps::NamedTuple) where {M, N, T}
	return d.σ.(mat_tensor_mult(ps.W, x))
end
