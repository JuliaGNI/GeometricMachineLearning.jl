@doc raw"""
    ForcingLayer <: AbstractExplicitLayer

Layers that can learn dissipative or forcing terms, but not conservative ones.

Use the constructors [`ForcingLayerQ`](@ref) and [`ForcingLayerP`](@ref) for this.
"""
struct ForcingLayer{M, N, PT<:Base.Callable, CT, type, ReturnParameters} <: AbstractExplicitLayer{M, N}
    dim::Int
    width::Int
    nhidden::Int
    parameter_length::Int
    parameter_convert::PT
    model::CT
end

function initialparameters(rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer, integrator::ForcingLayer, backend::KernelAbstractions.Backend, ::Type{T}) where {T}
    initialparameters(rng, init_weight, integrator.model, backend, T)
end

"""
    ForcingLayerQ

A layer that is derived from the more general [`ForcingLayer`](@ref) and only changes the ``q`` component.
"""
const ForcingLayerQ{M, N, FT, AT, ReturnParameters} = ForcingLayer{M, N, FT, AT, :Q, ReturnParameters}

"""
    ForcingLayerP

A layer that is derived from the more general [`ForcingLayer`](@ref) and only changes the ``p`` component.
"""
const ForcingLayerP{M, N, FT, AT, ReturnParameters} = ForcingLayer{M, N, FT, AT, :P, ReturnParameters}

function build_chain(dim::Integer, width::Integer, nhidden::Integer, parameter_length::Integer, activation)
    inner_layers = Tuple(
        [Dense(width, width, activation) for _ in 1:nhidden]
    )

    Chain(
        Dense(dim÷2 + parameter_length, width, activation),
        inner_layers...,
        Linear(width, dim÷2; use_bias = false)
    )
end

function ForcingLayer(dim::Integer, width::Integer, nhidden::Integer, activation; parameters::OptionalParameters, return_parameters::Bool, type::Symbol)
    flattened_parameters = ParameterHandling.flatten(parameters)
    parameter_length = length(flattened_parameters[1])
    c = build_chain(dim, width, nhidden, parameter_length, activation)
    ForcingLayer{dim, dim, typeof(flattened_parameters[2]), typeof(c), type, return_parameters}(dim, width, nhidden, parameter_length, flattened_parameters[2], c)
end

"""
    ForcingLayerQ(dim)

# Examples

```julia
ForcingLayerQ(dim, width, nhidden, activation; parameters, return_parameters)
```
"""
function ForcingLayerQ(dim::Integer, width::Integer=dim, nhidden::Integer=HNN_nhidden_default, activation=HNN_activation_default; parameters::OptionalParameters=NullParameters(), return_parameters::Bool=false)
    ForcingLayer(dim, width, nhidden, activation; parameters=parameters, return_parameters=return_parameters, type=:Q)
end

"""
    ForcingLayerP(dim)

See [`ForcingLayerQ`](@ref).

# Examples

```julia
ForcingLayerP(dim, width, nhidden, activation; parameters, return_parameters)
```
"""
function ForcingLayerP(dim::Integer, width::Integer=dim, nhidden::Integer=HNN_nhidden_default, activation=HNN_activation_default; parameters::OptionalParameters=NullParameters(), return_parameters::Bool=false)
    ForcingLayer(dim, width, nhidden, activation; parameters=parameters, return_parameters=return_parameters, type=:P)
end

function (integrator::ForcingLayerQ{M, N, FT, AT, false})(qp::QPT2, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.q, problem_params)
    (q = qp.q + integrator.model(input, params), p = qp.p)
end

function (integrator::ForcingLayerP{M, N, FT, AT, false})(qp::QPT2, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.p, problem_params)
    (q = qp.q, p = qp.p + integrator.model(input, params))
end

function (integrator::ForcingLayerQ{M, N, FT, AT, true})(qp::QPT2, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.q, problem_params)
    ((q = qp.q + integrator.model(input, params), p = qp.p), problem_params)
end

function (integrator::ForcingLayerP{M, N, FT, AT, true})(qp::QPT2, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.p, problem_params)
    ((q = qp.q, p = qp.p + integrator.model(input, params)), problem_params)
end

function (integrator::ForcingLayer)(qp_params::Tuple{<:QPTOAT2, <:OptionalParameters}, params::NeuralNetworkParameters)
    integrator(qp_params..., params)
end

function (integrator::ForcingLayer)(::TT, ::NeuralNetworkParameters) where {TT <: Tuple}
    error("The input is of type $(TT). This shouldn't be the case!")
end

function (integrator::ForcingLayer{M, N, FT, AT, Type, true})(qp::AbstractArray, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT, Type}
    @assert iseven(size(qp, 1))
    n = size(qp, 1)÷2
    qp_split = assign_q_and_p(qp, n)
    evaluated = integrator(qp_split, problem_params, params)[1]
    (vcat(evaluated.q, evaluated.p), problem_params)
end

function (integrator::ForcingLayer{M, N, FT, AT, Type, false})(qp::AbstractArray, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT, Type}
    @assert iseven(size(qp, 1))
    n = size(qp, 1)÷2
    qp_split = assign_q_and_p(qp, n)
    evaluated = integrator(qp_split, problem_params, params)
    vcat(evaluated.q, evaluated.p)
end

(integrator::ForcingLayer)(qp::QPTOAT2, params::NeuralNetworkParameters) = integrator(qp, NullParameters(), params)