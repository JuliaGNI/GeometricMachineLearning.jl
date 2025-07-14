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

    function SymbolicEnergy{Kinetic}(dim, width=dim, nhidden=HNN_nhidden_default, activation=HNN_activation_default; parameters::OptionalParameters=NullParameters()) where {Kinetic}
        @assert iseven(dim) "The input dimension must be an even integer!"
        flattened_parameters = flatten(parameters)
        parameter_length = length(flattened_parameters[2])
        new{typeof(activation), typeof(flattened_parameters[2]), Kinetic}(dim, width, nhidden, parameter_length, flattened_parameters[2], activation)
    end
end

ParameterHandling.flatten(::NullParameters) = flatten(NamedTuple())

"""
    SymbolicPotentialEnergy

# Constructors

```julia
SymbolicPotentialEnergy(dim)
```

# Parameter Dependence
"""
const SymbolicPotentialEnergy{AT, PT} = SymbolicEnergy{AT, PT, :potential}

"""
    SymbolicKineticEnergy

# Constructors

```julia

```
"""
const SymbolicKineticEnergy{AT, PT} = SymbolicEnergy{AT, PT, :kinetic}

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

function SymbolicNeuralNetworks.Jacobian(f::EqT, nn::AbstractSymbolicNeuralNetwork, dim2::Integer)
    # make differential 
    Dx = symbolic_differentials(nn.input)[1:dim2]

    # Evaluation of gradient
    s∇f = hcat([expand_derivatives.(Symbolics.scalarize(dx(f))) for dx in Dx]...)

    Jacobian(f, s∇f, nn)
end

function SymbolicNeuralNetworks.Jacobian(nn::AbstractSymbolicNeuralNetwork, dim2::Integer)
    
    # Evaluation of the symbolic output
    soutput = nn.model(nn.input, params(nn))

    Jacobian(soutput, nn, dim2)
end

function build_gradient(se::SymbolicEnergy)
    model = Chain(se)
    nn = SymbolicNeuralNetwork(model)
    □ = SymbolicNeuralNetworks.Jacobian(nn, se.dim÷2)
    SymbolicNeuralNetworks.build_nn_function(□, nn.params, nn.input)
end

struct SymplecticEuler{M, N, FT<:Callable, AT<:Architecture, type} <: AbstractExplicitLayer{M, N}
    gradient_function::FT
    energy_architecture::AT
end

function initialparameters(integrator::SymplecticEuler)

end

const SymplecticEulerA{M, N, FT, AT} = SymplecticEuler{M, N, FT, AT, :A}
const SymplecticEulerB{M, N, FT, AT} = SymplecticEuler{M, N, FT, AT, :B}

function SymplecticEulerA(se::SymbolicKineticEnergy)
    gradient_function = build_gradient(se)
    SymplecticEuler{build_gradient(se), :A}
end

function SymplecticEulerB(se::SymbolicPotentialEnergy)
    gradient_function = build_gradient(se)
    SymplecticEuler{build_gradient(se), :B}
end

(integrator::SymplecticEulerA)(qp::QPT, params::NeuralNetworkParameters) = (qp.q + integrator.gradient_function(qp.p, params), qp.p)
(integrator::SymplecticEulerB)(qp::QPT, params::NeuralNetworkParameters) = (qp.q, qp.p - integrator.gradient_function(qp.q, params))

SymbolicPotentialEnergy(args...; kwargs...) = SymbolicEnergy{:potential}(args...; kwargs...)
SymbolicKineticEnergy(args...; kwargs...) = KineticEnergy{:kinetic}(args...; kwargs...)

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
struct GeneralizedHamiltonianArchitecture{AT, IT} <: HamiltonianArchitecture{AT}
    dim::Int
    width::Int
    nhidden::Int
    n_integrators::Int
    activation::AT

    function GeneralizedHamiltonianArchitecture(dim, width=dim, nhidden=HNN_nhidden_default, n_integrators::Integer=1, activation=HNN_activation_default)
        new{typeof(activation)}(dim, width, nhidden, n_integrators, activation)
    end
end

function Chain(ghnn_arch::GeneralizedHamiltonianArchitecture)
    c = ()
    kinetic_energy = SymbolicKineticEnergy(ghnn_arch.dim, ghnn_arch.width, ghnn_arch.nhidden, ghnn_arch.parameter_length, ghnn_arch.parameter_convert, ghnn_arch.activation)
    potential_energy = SymbolicPotentialEnergy(ghnn_arch.dim, ghnn_arch.width, ghnn_arch.nhidden, ghnn_arch.parameter_length, ghnn_arch.parameter_convert, ghnn_arch.activation)
    
    for _ in 1:n_integrators
        c = (c..., SymplecticEulerA(kinetic_energy))
        c = (c..., SymplecticEulerB(potential_energy))
    end

    c
end

function HNNLoss(arch::GeneralizedHamiltonianArchitecture)

end