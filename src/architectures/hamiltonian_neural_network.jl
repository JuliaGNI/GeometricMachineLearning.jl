abstract type HamiltonianArchitecture{AT<:Activation} <: Architecture end

HNN_nhidden_default = 1
HNN_activation_default = tanh

function HamiltonianArchitecture(dim::Integer, width::Integer, nhidden::Integer, activation)
    @warn "You called the abstract type `HamiltonianArchitecture` as a constructor. This is defaulting to `StandardHamiltonianArchitecture`."
    StandardHamiltonianArchitecture(dim, width, nhidden, activation)
end

"""
    StandardHamiltonianArchitecture <: HamiltonianArchitecture

A realization of the standard Hamiltonian neural network (HNN) [greydanus2019hamiltonian](@cite).

Also see [`GeneralizedHamiltonianArchitecture`](@ref).

# Constructor

The constructor takes the following input arguments:
1. `dim`: system dimension,
2. `width = dim`: width of the hidden layer. By default this is equal to `dim`,
3. `nhidden = $(HNN_nhidden_default)`: the number of hidden layers,
4. `activation = $(HNN_activation_default)`: the activation function used in the HNN.
"""
struct StandardHamiltonianArchitecture{AT} <: HamiltonianArchitecture{AT}
    dim::Int
    width::Int
    nhidden::Int
    activation::AT

    function StandardHamiltonianArchitecture(dim, width=dim, nhidden=HNN_nhidden_default, activation=HNN_activation_default)
        new{typeof(activation)}(dim, width, nhidden, activation)
    end
end

GHNN_integrator_default = nothing

"""
    GeneralizedHamiltonianArchitecture <: HamiltonianArchitecture

A realization of generalized Hamiltonian neural networks (GHNNs) as introduced in [horn4555181generalized](@cite).

Also see [`StandardHamiltonianArchitecture`](@ref).

# Constructor

The constructor takes the following input arguments:
1. `dim`: system dimension,
2. `width = dim`: width of the hidden layer. By default this is equal to `dim`,
3. `nhidden = $(HNN_nhidden_default)`: the number of hidden layers,
4. `activation = $(HNN_activation_default)`: the activation function used in the GHNN,
5. `integrator = $(GHNN_integrator_default)`: the integrator that is used to design the GHNN.
"""
struct GeneralizedHamiltonianArchitecture{AT, IT} <: HamiltonianArchitecture{AT}
    dim::Int
    width::Int
    nhidden::Int
    activation::AT
    integrator::IT

    function GeneralizedHamiltonianArchitecture(dim, width=dim, nhidden=HNN_nhidden_default, activation=HNN_activation_default, integrator=GHNN_integrator_default)
        error("GHNN still has to be implemented!")
        new{typeof(activation), typeof(integrator)}(dim, width, nhidden, activation, integrator)
    end
end

@inline AbstractNeuralNetworks.dim(arch::HamiltonianArchitecture) = arch.dimin

"""
    symbolic_hamiltonian_vector_field(nn::SymbolicNeuralNetwork)

Get the symbolic expression for the vector field belonging to the HNN `nn`.

# Implementation 

This is calling `SymbolicNeuralNetworks.Jacobian` and then multiplies the result with a Poisson tensor.
"""
function symbolic_hamiltonian_vector_field(nn::SymbolicNeuralNetwork)
    □ = SymbolicNeuralNetworks.Jacobian(nn)
    input_dim = input_dimension(nn.model)
    n = input_dim ÷ 2
    # placeholder for one
    @variables o
    o_vec = repeat([o], n)
    𝕀 = Diagonal(o_vec)
    𝕆 = zero(𝕀)
    𝕁 = hcat(vcat(𝕆, -𝕀), vcat(𝕀, 𝕆))
    substitute(𝕁 * derivative(□)', Dict(o => 1, ))
end

"""
    hamiltonian_vector_field(arch::HamiltonianArchitecture)

Compute an executable expression of the Hamiltonian vector field of a [`HamiltonianArchitecture`](@ref).

# Implementation

This first computes a symbolic expression of the vector field using [`symbolic_hamiltonian_vector_field`](@ref).
"""
function hamiltonian_vector_field(arch::HamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    hvf = symbolic_hamiltonian_vector_field(nn)
    build_nn_function(hvf, nn.params, nn.input)
end

function Chain(arch::HamiltonianArchitecture)
    inner_layers = Tuple(
        [Dense(arch.width, arch.width, arch.activation) for _ in 1:arch.nhidden]
    )

    Chain(
        Dense(arch.dim, arch.width, arch.activation),
        inner_layers...,
        Linear(arch.width, 1; use_bias = false)
    )
end