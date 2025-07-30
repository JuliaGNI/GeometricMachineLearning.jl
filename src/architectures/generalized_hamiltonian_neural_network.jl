"""
    SymbolicEnergy

See [`SymbolicPotentialEnergy`](@ref) and [`SymbolicKineticEnergy`](@ref).
"""
struct SymbolicEnergy{AT <: Activation, PT, Kinetic} 
    dim::Int
    width::Int
    nhidden::Int
    parameter_length::Int
    parameter_convert::PT
    activation::AT

    function SymbolicEnergy(dim, width=dim, nhidden=HNN_nhidden_default, activation::Activation=HNN_activation_default; parameters::OptionalParameters=NullParameters(), type)
        @assert iseven(dim) "The input dimension must be an even integer!"
        flattened_parameters = ParameterHandling.flatten(parameters)
        parameter_length = length(flattened_parameters[1])
        new{typeof(activation), typeof(flattened_parameters[2]), type}(dim, width, nhidden, parameter_length, flattened_parameters[2], activation)
    end
end

ParameterHandling.flatten(::NullParameters) = ParameterHandling.flatten(NamedTuple())

"""
    SymbolicPotentialEnergy

A `const` derived from [`SymbolicEnergy`](@ref).

# Constructors

```jldoctest; setup=:(using GeometricMachineLearning)
julia> params, dim = (m = 1., ω = π / 2), 2
((m = 1.0, ω = 1.5707963267948966), 2)

julia> se = GeometricMachineLearning.SymbolicPotentialEnergy(dim; parameters = params);

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

function SymbolicNeuralNetworks.Jacobian(f::SymbolicNeuralNetworks.EqT, nn::AbstractSymbolicNeuralNetwork, dim2::Integer)
    # make differential 
    Dx = SymbolicNeuralNetworks.symbolic_differentials(nn.input)[1:dim2]

    # Evaluation of gradient
    s∇f = hcat([SymbolicNeuralNetworks.expand_derivatives.(SymbolicNeuralNetworks.Symbolics.scalarize(dx(f))) for dx in Dx]...)

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

```jldoctest; setup=:(using GeometricMachineLearning; using GeometricMachineLearning: SymbolicPotentialEnergy, build_gradient, concatenate_array_with_parameters; using GeometricMachineLearning.GeometricBase: OptionalParameters; using Random; Random.seed!(123))
params, dim = (m = 1., ω = π / 2), 4

se = SymbolicPotentialEnergy(dim; parameters = params)

network_params = NeuralNetwork(Chain(se)).params

built_grad = build_gradient(se)
grad(qp::AbstractArray, problem_params::OptionalParameters, params::NeuralNetworkParameters) = built_grad(concatenate_array_with_parameters(qp, problem_params), params)

grad(rand(2), params, network_params)

# output

2×1 Matrix{Float64}:
 0.069809944230798
 0.30150270218327446
```
"""
function build_gradient(se::SymbolicEnergy)
    model = Chain(se)
    nn = SymbolicNeuralNetwork(model)
    □ = SymbolicNeuralNetworks.Jacobian(nn, se.dim÷2)
    SymbolicNeuralNetworks.build_nn_function(SymbolicNeuralNetworks.derivative(□)', nn.params, nn.input)
end

struct SymplecticEuler{M, N, FT<:Base.Callable, MT<:Chain, type, Last} <: AbstractExplicitLayer{M, N}
    gradient_function::FT
    energy_model::MT
end

function initialparameters(rng::Random.AbstractRNG, init_weight::AbstractNeuralNetworks.Initializer, integrator::SymplecticEuler, backend::KernelAbstractions.Backend, ::Type{T}) where {T}
    initialparameters(rng, init_weight, integrator.energy_model, backend, T)
end

const SymplecticEulerA{M, N, FT, AT, Last} = SymplecticEuler{M, N, FT, AT, :A}
const SymplecticEulerB{M, N, FT, AT, Last} = SymplecticEuler{M, N, FT, AT, :B}

function SymplecticEulerA(se::SymbolicKineticEnergy; last_step::Bool=false)
    gradient_function = build_gradient(se)
    c = Chain(se)
    SymplecticEuler{se.dim, se.dim, typeof(gradient_function), typeof(c), :A, last_step}(gradient_function, c)
end

function SymplecticEulerB(se::SymbolicPotentialEnergy; last_step::Bool=false)
    gradient_function = build_gradient(se)
    c = Chain(se)
    SymplecticEuler{se.dim, se.dim, typeof(gradient_function), typeof(c), :B, last_step}(gradient_function, c)
end

function concatenate_array_with_parameters(qp::AbstractVector, params::OptionalParameters)
    vcat(qp, ParameterHandling.flatten(params)[1])
end

# function concatenate_array_with_parameters(qp::AbstractMatrix, params::OptionalParameters)
#     hcat((concatenate_array_with_parameters(qp[:, i], params) for i in axes(qp, 2))...)
# end

function concatenate_array_with_parameters(qp::AbstractMatrix, params::AbstractVector)
    @assert _size(qp, 2) == length(params)
    LazyArrays.Vcat((concatenate_array_with_parameters(@view(qp[:, i]), params[i]) for i in axes(params, 1))...)
end

function (integrator::SymplecticEulerA{M, N, FT, AT, true})(qp::QPT2, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.p, problem_params)
    (q = @view((qp.q + integrator.gradient_function(input, params))[:, 1]), p = qp.p)
end

function (integrator::SymplecticEulerB{M, N, FT, AT, true})(qp::QPT2, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.q, problem_params)
    (q = qp.q, p = @view((qp.p - integrator.gradient_function(input, params))[:, 1]))
end

function (integrator::SymplecticEulerA{M, N, FT, AT, false})(qp::QPT2, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.p, problem_params)
    ((q = @view((qp.q + integrator.gradient_function(input, params))[:, 1]), p = qp.p), problem_params)
end

function (integrator::SymplecticEulerB{M, N, FT, AT, false})(qp::QPT2, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT}
    input = concatenate_array_with_parameters(qp.q, problem_params)
    ((q = qp.q, p = @view((qp.p - integrator.gradient_function(input, params))[:, 1])), problem_params)
end

function (integrator::SymplecticEuler)(qp_params::Tuple{<:QPTOAT2, <:OptionalParameters}, params::NeuralNetworkParameters)
    integrator(qp_params..., params)
end

function (integrator::SymplecticEuler)(::TT, ::NeuralNetworkParameters) where {TT <: Tuple}
    error("The input is of type $(TT). This shouldn't be the case!")
end

function (integrator::SymplecticEuler{M, N, FT, AT, Type, false})(qp::AbstractArray, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT, Type}
    @assert iseven(size(qp, 1))
    n = size(qp, 1)÷2
    qp_split = assign_q_and_p(qp, n)
    evaluated = integrator(qp_split, problem_params, params)[1]
    (vcat(evaluated.q, evaluated.p), problem_params)
end

function (integrator::SymplecticEuler{M, N, FT, AT, Type, true})(qp::AbstractArray, problem_params::OptionalParameters, params::NeuralNetworkParameters) where {M, N, FT, AT, Type}
    @assert iseven(size(qp, 1))
    n = size(qp, 1)÷2
    qp_split = assign_q_and_p(qp, n)
    evaluated = integrator(qp_split, problem_params, params)
    vcat(evaluated.q, evaluated.p)
end

(integrator::SymplecticEuler)(qp::QPTOAT2, params::NeuralNetworkParameters) = integrator(qp, NullParameters(), params)

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
        new{typeof(activation), typeof(parameters)}(dim, width, nhidden, n_integrators, parameters, activation)
    end
end

@generated function AbstractNeuralNetworks.applychain(layers::Tuple, x::Tuple{<:QPTOAT2, <:OptionalParameters}, ps::Tuple)
    N = length(fieldtypes((layers)))
    x_symbols = vcat([:x], [gensym() for _ in 1:N])
    calls = [:(($(x_symbols[i + 1])) = layers[$i]($(x_symbols[i]), ps[$i])) for i in 1:N]
    push!(calls, :(return $(x_symbols[N + 1])))
    return Expr(:block, calls...)
end

index_qpt(qp::QPT2{T, 2}, i, j) where {T} = (q = qp.q[i, j], p = qp.p[i, j])
index_gpt(qp::QPT2{T, 3}, i, j, k) where {T} = (q = qp.q[i, j, k], p = qp.p[i, j, k])

function Chain(ghnn_arch::GeneralizedHamiltonianArchitecture)
    c = ()
    kinetic_energy = SymbolicKineticEnergy(ghnn_arch.dim, ghnn_arch.width, ghnn_arch.nhidden, ghnn_arch.activation; parameters=ghnn_arch.parameters)
    potential_energy = SymbolicPotentialEnergy(ghnn_arch.dim, ghnn_arch.width, ghnn_arch.nhidden, ghnn_arch.activation; parameters=ghnn_arch.parameters)
    
    for n in 1:ghnn_arch.n_integrators
        c = (c..., SymplecticEulerA(kinetic_energy; last_step = false))
        c = n == ghnn_arch.n_integrators ? (c..., SymplecticEulerB(potential_energy; last_step=true)) : (c..., SymplecticEulerB(potential_energy; last_step=false))
    end

    Chain(c...)
end

function (nn::NeuralNetwork{GT})(qp::QPTOAT2, problem_params::OptionalParameters) where {GT <: GeneralizedHamiltonianArchitecture}
    nn.model(qp, problem_params, params(nn))
end

function (model::Chain)(qp::QPTOAT2, problem_params::OptionalParameters, params::Union{NeuralNetworkParameters, NamedTuple})
    model((qp, problem_params), params)
end

function (c::Chain)(qp::QPT2{T, 3}, system_params::AbstractVector, ps::Union{NamedTuple, NeuralNetworkParameters})::QPT2{T} where {T}
    @assert size(qp.q, 3) == length(system_params)
    @assert size(qp.q, 2) == 1
    output_vectorwise = [c(index_gpt(qp, :, 1, i), system_params[i], ps) for i in axes(system_params, 1)]
    q_output = hcat([single_output_vectorwise.q for single_output_vectorwise ∈ output_vectorwise]...)
    p_output = hcat([single_output_vectorwise.p for single_output_vectorwise ∈ output_vectorwise]...)
    (q = reshape(q_output, size(q_output, 1), 1, size(q_output, 2)), p = reshape(p_output, size(p_output, 1), 1, size(p_output, 2)))
end