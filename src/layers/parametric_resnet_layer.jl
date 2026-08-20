@doc raw"""
    ParametricResNetLayer(dim, width, activation; parameters, return_parameters)

A [`WideResNetLayer`](@ref) whose hidden layer also sees the parameters of the *system*.

The flattened system parameters are appended to the input of the upscaling weight, so the layer
computes

```math
    x \mapsto x + \sigma(W_\mathrm{down}\sigma(W_\mathrm{up}[x; \mu] + b_\mathrm{up}) + b),
```

where ``\mu`` are the system parameters. `W_\mathrm{up}` is therefore
``\mathrm{width}\times(\mathrm{dim} + |\mu|)`` wide.

# Keyword arguments

- `parameters = NullParameters()`: a `NamedTuple` of system parameters, used only for its *shape* --
  the layer stores the resulting `NeuralNetworkParameters.ParameterLayout` and the flattened length,
  and the values are supplied per call.
- `return_parameters::Bool`: whether to pass the system parameters on to the next layer alongside the
  output, which is what lets a `Chain` of these thread them through.

This is the building block of [`ParametricResNet`](@ref).
"""
struct ParametricResNetLayer{M, N, F1 <: Activation, PT, ReturnParameters} <: AbstractExplicitLayer{M, N}
    width::Int
    activation::F1
    parameter_length::Int
    parameter_layout::PT
end

function ParametricResNetLayer(dim::Integer, width::Integer, activation=identity; parameters::OptionalParameters=NullParameters(), return_parameters::Bool)
    flat_parameters, layout = _flatten_system_parameters(parameters)
    _activation = Activation(activation)
    ParametricResNetLayer{dim, dim, typeof(_activation), typeof(layout), return_parameters}(width, _activation, length(flat_parameters), layout)
end

function initialparameters(rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer, l::ParametricResNetLayer{M, M}, backend::KernelAbstractions.Backend, ::Type{T}; init_bias = ZeroInitializer()) where {M, T}
    upscale_weight = KernelAbstractions.allocate(backend, T, l.width, M + l.parameter_length)
    upscale_bias = KernelAbstractions.allocate(backend, T, l.width)
    downscale_weight = KernelAbstractions.allocate(backend, T, M, l.width)
    bias = KernelAbstractions.allocate(backend, T, M)
    init_weight(rng, upscale_weight)
    init_weight(rng, downscale_weight)
    init_bias(rng, upscale_bias)
    init_bias(rng, bias)
    (upscale_weight=upscale_weight, downscale_weight=downscale_weight, upscale_bias=upscale_bias, bias=bias)
end

parameterlength(l::ParametricResNetLayer{M, M}) where {M} = (l.width + l.parameter_length) * (M + 1) + M * (l.width + 1)

function (d::ParametricResNetLayer{M, M, F, PT, false})(x::AbstractVecOrMat, problem_params::OptionalParameters, ps::NamedTuple) where {M, F, PT}
    input = concatenate_array_with_parameters(x, problem_params)
    x + d.activation.(ps.downscale_weight * d.activation.(ps.upscale_weight * input .+ ps.upscale_bias) .+ ps.bias)
end

function (d::ParametricResNetLayer{M, M, F, PT, true})(x::AbstractVecOrMat, problem_params::OptionalParameters, ps::NamedTuple) where {M, F, PT}
    input = concatenate_array_with_parameters(x, problem_params)
    (x + d.activation.(ps.downscale_weight * d.activation.(ps.upscale_weight * input .+ ps.upscale_bias) .+ ps.bias), problem_params)
end

(d::ParametricResNetLayer)(input::Tuple, ps::NamedTuple) = length(input) == 2 ? d(input..., ps) : error("The tuple must contain the input array/nt as well as the system parameters.")

function (d::ParametricResNetLayer{M, M, F, PT, false})(z::QPT, problem_params::OptionalParameters, ps::NamedTuple) where {M, F, PT}
    @assert iseven(M)
    @assert size(z.q, 1) * 2 == M
    N2 = M ÷ 2
    output = d(vcat(z.q, z.p), problem_params, ps)
    assign_q_and_p(output, N2)
end

function (d::ParametricResNetLayer{M, M, F, PT, true})(z::QPT, problem_params::OptionalParameters, ps::NamedTuple) where {M, F, PT}
    @assert iseven(M)
    @assert size(z.q, 1) * 2 == M
    N2 = M ÷ 2
    output = d(vcat(z.q, z.p), problem_params, ps)
    (assign_q_and_p(output[1], N2), problem_params)
end