"""
    HamiltonianArchitecture <: Architecture

See [`StandardHamiltonianArchitecture`](@ref) and [`GeneralizedHamiltonianArchitecture`](@ref).
"""
abstract type HamiltonianArchitecture{AT<:Activation} <: Architecture end

const HNN_nhidden_default = 1
const HNN_activation_default = AbstractNeuralNetworks.TanhActivation()

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

A realization of generalized Hamiltonian neural networks (GHNNs) as introduced in [horn2025generalized](@cite).

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

@inline AbstractNeuralNetworks.dim(arch::HamiltonianArchitecture) = arch.dim

"""
    symbolic_hamiltonian_vector_field(nn::SymbolicNeuralNetwork)

Get the symbolic expression for the vector field belonging to the HNN `nn`.

# Implementation 

This is calling `SymbolicNeuralNetworks.Jacobian` and then multiplies the result with a Poisson tensor.
"""
function symbolic_hamiltonian_vector_field(nn::SymbolicNeuralNetwork)
    □ = SymbolicNeuralNetworks.Jacobian(nn)
    n = input_dimension(nn.model) ÷ 2
    # The Poisson tensor is built from *integers* on purpose: a `Float64` literal in the symbolic
    # expression would widen the result of a `Float32` network. `PoissonTensor`, which has the same
    # convention, is an `AbstractMatrix{Float64}` and would do exactly that.
    𝕆 = zeros(Int, n, n)
    𝕀 = Matrix(1I, n, n)
    𝕁 = [𝕆 𝕀; -𝕀 𝕆]
    # `Jacobian` uses the convention `□[i, j] = ∂fᵢ/∂xⱼ` and the HNN output is scalar, so the one
    # row of `derivative(□)` is the gradient of the Hamiltonian. The vector field is built as a
    # *vector*, so that the generated function returns what `HNNLoss` compares against: a vector
    # for a single sample, and one column per sample for a batch.
    ∇H = vec(derivative(□))
    𝕁 * ∇H
end

"""
    hamiltonian_vector_field(arch::HamiltonianArchitecture)

Compute an executable expression of the Hamiltonian vector field of a [`HamiltonianArchitecture`](@ref).

# Implementation

This first computes a symbolic expression of the vector field using [`symbolic_hamiltonian_vector_field`](@ref).

The function is built with `inplace = false`: [`HNNLoss`](@ref) wraps it and is differentiated with
`Zygote`, and the in-place kernel `SymbolicNeuralNetworks.build_nn_function` builds by default
*mutates* its result, which `Zygote` does not support.
"""
function hamiltonian_vector_field(arch::HamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    hvf = symbolic_hamiltonian_vector_field(nn)
    SymbolicNeuralNetworks.build_nn_function(hvf, nn.params, nn.input; inplace = false)
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