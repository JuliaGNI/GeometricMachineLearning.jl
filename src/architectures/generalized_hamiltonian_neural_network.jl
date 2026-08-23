"""
    SymbolicEnergy

See [`SymbolicPotentialEnergy`](@ref) and [`SymbolicKineticEnergy`](@ref).
"""
struct SymbolicEnergy{AT <: Activation, PT, Kinetic} 
    dim::Int
    width::Int
    nhidden::Int
    parameter_length::Int
    parameter_layout::PT
    activation::AT

    function SymbolicEnergy(dim, width, nhidden, activation; parameters::OptionalParameters=NullParameters(), type)
        @assert iseven(dim) "The input dimension must be an even integer!"
        flat_parameters, layout = _flatten_system_parameters(parameters)
        _activation = Activation(activation)
        new{typeof(_activation), typeof(layout), type}(dim, width, nhidden, length(flat_parameters), layout, _activation)
    end
end

"""
    SymbolicPotentialEnergy

A `const` derived from [`SymbolicEnergy`](@ref).

# Constructors

```jldoctest; setup=:(using GeometricMachineLearning; using GeometricMachineLearning: Activation)
julia> params, dim, width, nhidden, activation = (m = 1., ω = π / 2), 2, 2, 1, tanh
((m = 1.0, ω = 1.5707963267948966), 2, 2, 1, tanh)

julia> se = GeometricMachineLearning.SymbolicPotentialEnergy(dim, width, nhidden, activation; parameters = params);

```

In practice we use `SymbolicPotentialEnergy` (and [`SymbolicKineticEnergy`](@ref)) together with [`build_gradient(::SymbolicEnergy)`](@ref).

# Parameter Dependence
"""
const SymbolicPotentialEnergy{AT, PT} = SymbolicEnergy{AT, PT, :potential}

"""
    SymbolicKineticEnergy

A `const` derived from [`SymbolicEnergy`](@ref).

# Constructors

See [`SymbolicPotentialEnergy`](@ref).
"""
const SymbolicKineticEnergy{AT, PT} = SymbolicEnergy{AT, PT, :kinetic}

SymbolicPotentialEnergy(args...; kwargs...) = SymbolicEnergy(args...; type = :potential, kwargs...)
SymbolicKineticEnergy(args...; kwargs...) = SymbolicEnergy(args...; type = :kinetic, kwargs...)

function Chain(se::SymbolicEnergy)
    inner_layers = Tuple(
        [Dense(se.width, se.width, se.activation) for _ in 1:se.nhidden]
    )

    Chain(
        Dense(se.dim÷2 + se.parameter_length, se.width, se.activation),
        inner_layers...,
        Linear(se.width, 1; use_bias = false)
    )
end

# Jacobian with respect to the *first* `dim2` input variables only: for a parameter-dependent
# network the remaining input components are the system parameters, which are not differentiated.
#
# TODO: type piracy -- `Jacobian` and `AbstractSymbolicNeuralNetwork` are both
# SymbolicNeuralNetworks'. The restricted-Jacobian variant belongs there.
function SymbolicNeuralNetworks.Jacobian(f, nn::AbstractSymbolicNeuralNetwork, dim2::Integer)
    # make differential of input variables (not of parameters)
    Dx = SymbolicNeuralNetworks.symbolic_differentials(nn.input)[1:dim2]

    # Evaluation of gradient
    s∇f = hcat([SymbolicNeuralNetworks.expand_derivatives.(dx.(SymbolicNeuralNetworks.Symbolics.scalarize(f))) for dx in Dx]...)

    SymbolicNeuralNetworks.Jacobian(f, s∇f, nn)
end

function SymbolicNeuralNetworks.Jacobian(nn::AbstractSymbolicNeuralNetwork, dim2::Integer)
    
    # Evaluation of the symbolic output
    soutput = nn.model(nn.input, params(nn))

    SymbolicNeuralNetworks.Jacobian(soutput, nn, dim2)
end

"""
    build_gradient(se)

Build a gradient function from a [`SymbolicEnergy`](@ref) `se`.

# Examples

```jldoctest; setup=:(using GeometricMachineLearning; using GeometricMachineLearning: SymbolicPotentialEnergy, build_gradient, concatenate_array_with_parameters, OneInitializer; using GeometricMachineLearning.GeometricBase: OptionalParameters)
params, dim, width, nhidden, activation = (m = 1., ω = π / 2), 4, 2, 1, tanh

se = SymbolicPotentialEnergy(dim, width, nhidden, activation; parameters = params)

# `OneInitializer` rather than the default random one, so that the output below does not depend on
# the random number stream of the Julia version
network_params = NeuralNetwork(Chain(se); initializer = OneInitializer()).params

built_grad = build_gradient(se)
grad(qp::AbstractArray, problem_params::OptionalParameters, params::NetworkParameters) = built_grad(concatenate_array_with_parameters(qp, problem_params), params)

grad([0.5, 0.25], params, network_params)

# output

2×1 Matrix{Float64}:
 2.7907683385233434e-5
 2.7907683385233434e-5
```
"""
function build_gradient(se::SymbolicEnergy)
    model = Chain(se)
    nn = SymbolicNeuralNetwork(model)
    □ = SymbolicNeuralNetworks.Jacobian(nn, se.dim÷2)
    SymbolicNeuralNetworks.build_nn_function(SymbolicNeuralNetworks.derivative(□)', nn.params, nn.input;
                                             inplace = false)
end

struct SymplecticEuler{M, N, FT<:Base.Callable, MT<:Chain, type, ReturnParameters} <: AbstractExplicitLayer{M, N}
    gradient_function::FT
    energy_model::MT
end

function parameterlength(integrator::SymplecticEuler)
    parameterlength(integrator.energy_model)
end

function initialparameters(rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer, integrator::SymplecticEuler, backend::KernelAbstractions.Backend, ::Type{T}) where {T}
    initialparameters(rng, init_weight, integrator.energy_model, backend, T)
end

const SymplecticEulerA{M, N, FT, AT, ReturnParameters} = SymplecticEuler{M, N, FT, AT, :A, ReturnParameters}
const SymplecticEulerB{M, N, FT, AT, ReturnParameters} = SymplecticEuler{M, N, FT, AT, :B, ReturnParameters}

"""
Changes ``q`` (based on the kinetic energy).
"""
function SymplecticEulerA(se::SymbolicKineticEnergy; return_parameters::Bool)
    gradient_function = build_gradient(se)
    c = Chain(se)
    SymplecticEuler{se.dim, se.dim, typeof(gradient_function), typeof(c), :A, return_parameters}(gradient_function, c)
end

"""
Changes ``p`` (based on the potential energy).
"""
function SymplecticEulerB(se::SymbolicPotentialEnergy; return_parameters::Bool)
    gradient_function = build_gradient(se)
    c = Chain(se)
    SymplecticEuler{se.dim, se.dim, typeof(gradient_function), typeof(c), :B, return_parameters}(gradient_function, c)
end

# A network with no system parameters gets its input unchanged; without this the empty flat vector
# would have to be `vcat`ed on, which loses the element type.
concatenate_array_with_parameters(qp::AbstractArray, ::NullParameters) = qp

function concatenate_array_with_parameters(qp::AbstractVector, params::NamedTuple)
    vcat(qp, _flatten_system_parameters(params)[1])
end

function concatenate_array_with_parameters(qp::AbstractMatrix, params::NamedTuple)
    @assert size(qp, 2) == 1
    vcat(qp, repeat(_flatten_system_parameters(params)[1], 1, size(qp, 2)))
end

function concatenate_array_with_parameters(qp::AbstractArray{T, 3}, params::AbstractVector) where {T}
    @assert size(qp, 3) == length(params)
    matrices = Tuple(concatenate_array_with_parameters(qp[:, :, i], params[i]) for i in axes(qp, 3))
    cat(matrices...; dims = 3)
end

# function concatenate_array_with_parameters(qp::AbstractMatrix, params::OptionalParameters)
#     hcat((concatenate_array_with_parameters(qp[:, i], params) for i in axes(qp, 2))...)
# end

# One parameter set per column, so the columns are concatenated *side by side*: the result is a
# matrix with `size(qp, 1) + parameter_length` rows, one column per sample.
function concatenate_array_with_parameters(qp::AbstractMatrix, params::AbstractVector)
    @assert _size(qp, 2) == length(params)
    hcat((concatenate_array_with_parameters(@view(qp[:, i]), params[i]) for i in axes(params, 1))...)
end

function (integrator::SymplecticEulerA{M, N, FT, AT, false})(qp::QPT2, problem_params::OptionalParameters, params::NetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.p, problem_params)
    (q = @view((qp.q + integrator.gradient_function(input, params))[:, 1]), p = qp.p)
end

function (integrator::SymplecticEulerB{M, N, FT, AT, false})(qp::QPT2, problem_params::OptionalParameters, params::NetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.q, problem_params)
    (q = qp.q, p = @view((qp.p - integrator.gradient_function(input, params))[:, 1]))
end

function (integrator::SymplecticEulerA{M, N, FT, AT, true})(qp::QPT2, problem_params::OptionalParameters, params::NetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.p, problem_params)
    ((q = @view((qp.q + integrator.gradient_function(input, params))[:, 1]), p = qp.p), problem_params)
end

function (integrator::SymplecticEulerB{M, N, FT, AT, true})(qp::QPT2, problem_params::OptionalParameters, params::NetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.q, problem_params)
    ((q = qp.q, p = @view((qp.p - integrator.gradient_function(input, params))[:, 1])), problem_params)
end

function (integrator::SymplecticEuler)(qp_params::Tuple{<:QPTOAT2, <:OptionalParameters}, params::NetworkParameters)
    integrator(qp_params..., params)
end

function (integrator::SymplecticEuler)(::TT, ::NetworkParameters) where {TT <: Tuple}
    error("The input is of type $(TT). This shouldn't be the case!")
end

function (integrator::SymplecticEuler{M, N, FT, AT, Type, true})(qp::AbstractArray, problem_params::OptionalParameters, params::NetworkParameters) where {M, N, FT, AT, Type}
    @assert iseven(size(qp, 1))
    n = size(qp, 1)÷2
    qp_split = assign_q_and_p(qp, n)
    evaluated = integrator(qp_split, problem_params, params)[1]
    (vcat(evaluated.q, evaluated.p), problem_params)
end

function (integrator::SymplecticEuler{M, N, FT, AT, Type, false})(qp::AbstractArray, problem_params::OptionalParameters, params::NetworkParameters) where {M, N, FT, AT, Type}
    @assert iseven(size(qp, 1))
    n = size(qp, 1)÷2
    qp_split = assign_q_and_p(qp, n)
    evaluated = integrator(qp_split, problem_params, params)
    vcat(evaluated.q, evaluated.p)
end

(integrator::SymplecticEuler)(qp::QPTOAT2, params::NetworkParameters) = integrator(qp, NullParameters(), params)

"""
    GeneralizedHamiltonianArchitecture <: HamiltonianArchitecture

A realization of generalized Hamiltonian neural networks (GHNNs) as introduced in [horn2025generalized](@cite).

Also see [`StandardHamiltonianArchitecture`](@ref).

# Constructor

The constructor takes the following input arguments:
1. `dim`: system dimension,
2. `width = dim`: width of the hidden layer. By default this is equal to `dim`,
3. `nhidden = $(HNN_nhidden_default)`: the number of hidden layers,
4. `n_integrators`: the number of integrators used in the GHNN.
5. `activation = $(HNN_activation_default)`: the activation function used in the GHNN,
"""
struct GeneralizedHamiltonianArchitecture{AT, PT <: OptionalParameters} <: HamiltonianArchitecture{AT}
    dim::Int
    width::Int
    nhidden::Int
    n_integrators::Int
    parameters::PT
    activation::AT

    function GeneralizedHamiltonianArchitecture(dim; width=dim, nhidden=HNN_nhidden_default, n_integrators::Integer=1, activation=HNN_activation_default, parameters=NullParameters())
        activation = (typeof(activation) <: Activation) ? activation : Activation(activation)
        new{typeof(activation), typeof(parameters)}(dim, width, nhidden, n_integrators, parameters, activation)
    end
end

index_qpt(qp::QPT2{T, 2}, i, j) where {T} = (q = qp.q[i, j], p = qp.p[i, j])
index_gpt(qp::QPT2{T, 3}, i, j, k) where {T} = (q = qp.q[i, j, k], p = qp.p[i, j, k])

function Chain(ghnn_arch::GeneralizedHamiltonianArchitecture)
    c = ()
    kinetic_energy = SymbolicKineticEnergy(ghnn_arch.dim, ghnn_arch.width, ghnn_arch.nhidden, ghnn_arch.activation; parameters=ghnn_arch.parameters)
    potential_energy = SymbolicPotentialEnergy(ghnn_arch.dim, ghnn_arch.width, ghnn_arch.nhidden, ghnn_arch.activation; parameters=ghnn_arch.parameters)
    
    for n in 1:ghnn_arch.n_integrators
        c = (c..., SymplecticEulerA(kinetic_energy; return_parameters = true))
        c = n == ghnn_arch.n_integrators ? (c..., SymplecticEulerB(potential_energy; return_parameters=false)) : (c..., SymplecticEulerB(potential_energy; return_parameters=true))
    end

    Chain(c...)
end

function (nn::NeuralNetwork{GT})(qp::QPTOAT2, problem_params::OptionalParameters) where {GT <: GeneralizedHamiltonianArchitecture}
    apply_parametric(nn.model, qp, problem_params, params(nn))
end

@doc raw"""
    apply_parametric(model, qp, system_parameters, ps)

Apply `model` to `qp` with the parameters of the *system* alongside the state, and the network
parameters `ps`.

`system_parameters` is either one `OptionalParameters` for the whole input, or — for a batch drawn by
[`ParametricDataLoader`](@ref) — one entry per sample.

# Implementation

A `Chain` of parameter-dependent layers threads `(state, system parameters)` from layer to layer, so
this hands the pair to the ordinary two-argument `Chain` functor and lets
`AbstractNeuralNetworks.applychain` carry it: since AbstractNeuralNetworks 0.7 that method leaves its
data argument untyped, so a tuple needs no method of its own.

This is a function of GML's rather than a three-argument functor on `Chain`, which would be type
piracy — `Chain` and every argument type belong to AbstractNeuralNetworks.
"""
function apply_parametric(model::Chain, qp::QPTOAT2, problem_params::OptionalParameters,
        ps::Union{NetworkParameters, NamedTuple})
    model((qp, problem_params), ps)
end

function apply_parametric(c::Chain, qp::QPT2{T, 3}, system_params::AbstractVector,
        ps::Union{NamedTuple, NetworkParameters})::QPT2{T} where {T}
    @assert size(qp.q, 3) == length(system_params)
    @assert size(qp.q, 2) == 1
    output_vectorwise = [apply_parametric(c, index_gpt(qp, :, 1, i), system_params[i], ps)
                         for i in axes(system_params, 1)]
    q_output = hcat([single_output_vectorwise.q for single_output_vectorwise ∈ output_vectorwise]...)
    p_output = hcat([single_output_vectorwise.p for single_output_vectorwise ∈ output_vectorwise]...)
    (q = reshape(q_output, size(q_output, 1), 1, size(q_output, 2)),
     p = reshape(p_output, size(p_output, 1), 1, size(p_output, 2)))
end

function apply_parametric(c::Chain, qp::AbstractArray{T, 2}, system_params::AbstractVector,
        ps::Union{NamedTuple, NetworkParameters}) where {T}
    @assert _size(qp, 2) == length(system_params)
    qp_reshaped = reshape(qp, size(qp, 1), 1, length(system_params))
    apply_parametric(c, qp_reshaped, system_params, ps)
end

function apply_parametric(c::Chain, qp::AbstractArray{T, 3}, system_params::AbstractVector,
        ps::Union{NamedTuple, NetworkParameters}) where {T}
    @assert size(qp, 3) == length(system_params)
    @assert size(qp, 2) == 1
    @assert iseven(size(qp, 1))
    n = size(qp, 1)÷2
    qp_split = assign_q_and_p(qp, n)
    c_output = apply_parametric(c, qp_split, system_params, ps)::QPT
    reshape(vcat(c_output.q, c_output.p), 2n, length(system_params))
end
