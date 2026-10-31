@doc raw"""
    ParametricResNetLayer(dim, width, activation; parameters, return_parameters)

A [`WideResNetLayer`](@ref) whose hidden layer also sees the parameters of the *system*.

The flattened system parameters are appended to the input of the upscaling weight, so the layer
computes

```math
    x \mapsto x + \sigma(W_\mathrm{down}\sigma(W_\mathrm{up}[x; \mu] + b_\mathrm{up}) + b),
```

where ``\mu`` are the system parameters. ``W_\mathrm{up}`` is therefore
``\mathrm{width}\times(\mathrm{dim} + |\mu|)`` wide, and without system parameters the map is that
of a [`WideResNetLayer`](@ref) with the same width and weights. The input is a vector, a matrix of
samples, a tensor of samples, or a `(q, p)` pair of any of these, with one `μ` — the forms that
[`WideResNetLayer`](@ref) takes.

# Keyword arguments

- `parameters = NullParameters()`: a `NamedTuple` of system parameters, used only for its *shape* —
  the layer stores the resulting `NeuralNetworkParameters.ParameterLayout` and the flattened length,
  and the values are supplied per call.
- `return_parameters = false`: whether to pass the system parameters on to the next layer beside the
  output, which is what lets a `Chain` of these thread them through. A `Chain` of one layer only
  takes the default.

This is the building block of [`ParametricResNet`](@ref).
"""
struct ParametricResNetLayer{M, N, F1, PT, ReturnParameters} <: AbstractExplicitLayer{M, N}
    width::Int
    activation::F1
    parameter_length::Int
    parameter_layout::PT
end

function ParametricResNetLayer(dim::Integer, width::Integer, activation = identity;
        parameters::OptionalParameters = NullParameters(), return_parameters::Bool = false)
    flat_parameters, layout = flatten_system_parameters(parameters)
    ParametricResNetLayer{dim, dim, typeof(activation), typeof(layout), return_parameters}(
        width, activation, length(flat_parameters), layout)
end

function initialparameters(
        rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer,
        l::ParametricResNetLayer{M, M}, backend::KernelAbstractions.Backend,
        ::Type{T}; init_bias = ZeroInitializer()) where {M, T}
    upscale_weight = KernelAbstractions.allocate(
        backend, T, l.width, M + l.parameter_length)
    upscale_bias = KernelAbstractions.allocate(backend, T, l.width)
    downscale_weight = KernelAbstractions.allocate(backend, T, M, l.width)
    bias = KernelAbstractions.allocate(backend, T, M)
    init_weight(rng, upscale_weight)
    init_weight(rng, downscale_weight)
    init_bias(rng, upscale_bias)
    init_bias(rng, bias)
    (upscale_weight = upscale_weight, downscale_weight = downscale_weight,
        upscale_bias = upscale_bias, bias = bias)
end

# The upscaling weight, its bias, the downscaling weight and the bias: the four arrays that
# `initialparameters` allocates.
function parameterlength(l::ParametricResNetLayer{M, M}) where {M}
    l.width * (M + l.parameter_length) + l.width + M * (l.width + 1)
end

function (d::ParametricResNetLayer{M, M, F, PT, false})(
        x::AbstractArray, μ::OptionalParameters, ps::NamedTuple) where {M, F, PT}
    input = concatenate_array_with_parameters(x, μ)
    x + d.activation.(_mul(ps.downscale_weight,
        d.activation.(_mul(ps.upscale_weight, input) .+ ps.upscale_bias)) .+ ps.bias)
end

function (d::ParametricResNetLayer{M, M, F, PT, true})(
        x::AbstractArray, μ::OptionalParameters, ps::NamedTuple) where {M, F, PT}
    input = concatenate_array_with_parameters(x, μ)
    (
        x + d.activation.(_mul(ps.downscale_weight,
            d.activation.(_mul(ps.upscale_weight, input) .+ ps.upscale_bias)) .+ ps.bias),
        μ)
end

# The `Chain` hands the layer the pair of its input and the system parameters.
function (d::ParametricResNetLayer)(
        input::Tuple{<:Union{AbstractArray, NamedTuple}, <:OptionalParameters},
        ps::NamedTuple)
    d(input..., ps)
end

function (d::ParametricResNetLayer{M, M, F, PT, false})(
        z::NamedTupleOfArrays{<:Any, (:q, :p)}, μ::OptionalParameters,
        ps::NamedTuple) where {M, F, PT}
    @assert iseven(M)
    @assert size(z.q, 1) * 2 == M
    N2 = M ÷ 2
    output = d(vcat(z.q, z.p), μ, ps)
    assign_q_and_p(output, N2)
end

function (d::ParametricResNetLayer{M, M, F, PT, true})(
        z::NamedTupleOfArrays{<:Any, (:q, :p)}, μ::OptionalParameters,
        ps::NamedTuple) where {M, F, PT}
    @assert iseven(M)
    @assert size(z.q, 1) * 2 == M
    N2 = M ÷ 2
    output = d(vcat(z.q, z.p), μ, ps)
    (assign_q_and_p(output[1], N2), output[2])
end
