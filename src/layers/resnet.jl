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
struct ResNetLayer{M, N, bias, F1} <: AbstractExplicitLayer{M, N}
    # The bias is the type parameter `:bias` or `:no_bias`, as in `VolumePreservingFeedForwardLayer`.
    activation::F1
end

function ResNetLayer(dim::Integer, activation = identity; use_bias::Bool = true)
    bias = use_bias ? :bias : :no_bias
    ResNetLayer{dim, dim, bias, typeof(activation)}(activation)
end

function initialparameters(
        rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        ::ResNetLayer{M, M, :bias}, backend::KernelAbstractions.Backend,
        ::Type{T}; init_bias = ZeroInitializer()) where {M, T}
    weight = KernelAbstractions.allocate(backend, T, M, M)
    bias = KernelAbstractions.allocate(backend, T, M)
    init_weight(rng, weight)
    init_bias(rng, bias)
    (weight = weight, bias = bias)
end

function initialparameters(
        rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        ::ResNetLayer{M, M, :no_bias}, backend::KernelAbstractions.Backend,
        ::Type{T}; init_bias = ZeroInitializer()) where {M, T}
    weight = KernelAbstractions.allocate(backend, T, M, M)
    init_weight(rng, weight)
    (weight = weight,)
end

parameterlength(::ResNetLayer{M, M, :bias}) where {M} = M * (M + 1)

parameterlength(::ResNetLayer{M, M, :no_bias}) where {M} = M * M

function (d::ResNetLayer{M, M, :bias})(x::AbstractArray, ps::NamedTuple) where {M}
    x + d.activation.(_mul(ps.weight, x) .+ ps.bias)
end

function (d::ResNetLayer{M, M, :no_bias})(x::AbstractArray, ps::NamedTuple) where {M}
    x + d.activation.(_mul(ps.weight, x))
end

function (d::Union{Dense{M, N}, ResNetLayer{M, N}})(z::QPT, ps::NamedTuple) where {M, N}
    @assert iseven(M) == iseven(N) == true
    @assert size(z.q, 1) * 2 == M
    N2 = N ÷ 2
    output = d(vcat(z.q, z.p), ps)
    assign_q_and_p(output, N2)
end
