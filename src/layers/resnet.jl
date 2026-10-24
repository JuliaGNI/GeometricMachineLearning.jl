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
    activation::F1
end

function ResNetLayer(dim::Integer, activation = identity; use_bias::Bool = true)
    ResNetLayer{dim, dim, use_bias ? :bias : :no_bias, typeof(activation)}(activation)
end

# The bias of a layer whose bias parameter is `:bias` or `:no_bias`: a vector of length `M` filled
# by `init_bias`, or an empty `NamedTuple`. A layer merges it with its weight, which it draws first.
function _bias_parameters(::Val{:no_bias}, rng, init_bias, backend, ::Type{T}, M) where {T}
    NamedTuple()
end
function _bias_parameters(::Val{:bias}, rng, init_bias, backend, ::Type{T}, M) where {T}
    bias = KernelAbstractions.allocate(backend, T, M)
    init_bias(rng, bias)
    (bias = bias,)
end

function initialparameters(
        rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        ::ResNetLayer{M, M, bias}, backend::KernelAbstractions.Backend,
        ::Type{T}; init_bias = ZeroInitializer()) where {M, bias, T}
    weight = KernelAbstractions.allocate(backend, T, M, M)
    init_weight(rng, weight)
    merge((weight = weight,), _bias_parameters(Val(bias), rng, init_bias, backend, T, M))
end

parameterlength(::ResNetLayer{M, M, :bias}) where {M} = M * (M + 1)
parameterlength(::ResNetLayer{M, M, :no_bias}) where {M} = M * M

function (d::ResNetLayer{M, M, :bias})(x::AbstractArray, ps::NamedTuple) where {M}
    x + d.activation.(_mul(ps.weight, x) .+ ps.bias)
end

function (d::ResNetLayer{M, M, :no_bias})(x::AbstractArray, ps::NamedTuple) where {M}
    x + d.activation.(_mul(ps.weight, x))
end

function (d::Union{Dense{M, N}, ResNetLayer{M, N}})(
        z::NamedTupleOfArrays{<:Any, (:q, :p)}, ps::NamedTuple) where {M, N}
    @assert iseven(M) == iseven(N) == true
    @assert size(z.q, 1) * 2 == M
    N2 = N ÷ 2
    output = d(vcat(z.q, z.p), ps)
    assign_q_and_p(output, N2)
end
