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

    function StandardHamiltonianArchitecture(dim::Integer, width=dim, nhidden=HNN_nhidden_default, activation=HNN_activation_default)
        @assert iseven(dim) "The input dimension must be an even integer"
        new{typeof(activation)}(dim, width, nhidden, activation)
    end
end

@inline AbstractNeuralNetworks.dim(arch::HamiltonianArchitecture) = arch.dim

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
function hamiltonian_vector_field(arch::StandardHamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    hvf = symbolic_hamiltonian_vector_field(nn)
    SymbolicNeuralNetworks.build_nn_function(hvf, nn.params, nn.input)
end

function Chain(arch::StandardHamiltonianArchitecture)
    inner_layers = Tuple(
        [Dense(arch.width, arch.width, arch.activation) for _ in 1:arch.nhidden]
    )

    Chain(
        Dense(arch.dim, arch.width, arch.activation),
        inner_layers...,
        Linear(arch.width, 1; use_bias = false)
    )
end