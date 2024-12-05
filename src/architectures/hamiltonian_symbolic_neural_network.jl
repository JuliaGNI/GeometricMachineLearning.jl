"""
    HamiltonianSymbolicNeuralNetwork <: AbstractSymbolicNeuralNetwork

A struct that inherits properties from the abstract type `AbstractSymbolicNeuralNetwork`.

# Constructor

    HamiltonianSymbolicNeuralNetwork(model)

Make an instance of `HamiltonianSymbolicNeuralNetwork` based on a `Chain` or an `Architecture`.
This is similar to the constructor for [`SymbolicNeuralNetwork`](@ref) but also checks if the input dimension is even-dimensional and the output dimension is one.
"""
struct HamiltonianSymbolicNeuralNetwork{AT, MT, PT} <: AbstractSymbolicNeuralNetwork{AT}
    architecture::AT
    model::MT
    params::PT
end

function HamiltonianSymbolicNeuralNetwork(arch::Architecture, model::Model)
    @assert iseven(input_dimension(model)) "Input dimension has to be an even number."
    @assert output_dimension(model) == 1 "Output dimension of network has to be scalar."

    sparams = symbolicparameters(model)
    HamiltonianSymbolicNeuralNetwork(arch, model, sparams)
end

HamiltonianSymbolicNeuralNetwork(model::Model) = HamiltonianSymbolicNeuralNetwork(UnknownArchitecture(), model)
HamiltonianSymbolicNeuralNetwork(arch::Architecture) = HamiltonianSymbolicNeuralNetwork(arch, Chain(model))

"""
    vector_field(nn::HamiltonianSymbolicNeuralNetwork)

Get the symbolic expression for the vector field belonging to the HNN `nn`.

# Implementation 

This is calling [`SymbolicNeuralNetworks.Jacobian`](@ref) and then multiplies the result with a Poisson tensor.
"""
function vector_field(nn::HamiltonianSymbolicNeuralNetwork)
    gradient_output = gradient(nn)
    sinput, soutput, ∇nn = gradient_output.x, gradient_output.soutput, gradient_output.s∇output
    input_dim = input_dimension(nn.model)
    n = input_dim ÷ 2
    # placeholder for one
    @variables o
    o_vec = repeat([o], n)
    𝕀 = Diagonal(o_vec)
    𝕆 = zero(𝕀)
    𝕁 = hcat(vcat(𝕆, -𝕀), vcat(𝕀, 𝕆))
    (x = sinput, nn = soutput, ∇nn = ∇nn, hvf = substitute(𝕁 * ∇nn, Dict(o => 1, )))
end

"""
    HNNLoss <: NetworkLoss

The loss for a Hamiltonian neural network.

# Constructor

This can be called with an instance of [`HamiltonianSymbolicNeuralNetwork`](@ref) as the only input arguemtn, i.e.:
```julia
HNNLoss(nn)
```
where `nn` is a [`HamiltonianSymbolicNeuralNetwork`](@ref) gives the corresponding Hamiltonian loss.

# Funktor

```julia
loss(c, ps, input, output)
loss(ps, input, output) # equivalent to the above
```
"""
struct HNNLoss{FT} <: NetworkLoss
    hvf::FT
end

function HNNLoss(nn::HamiltonianSymbolicNeuralNetwork)
    x_hvf = vector_field(nn)
    x = x_hvf.x
    hvf = x_hvf.hvf
    hvf_function = build_nn_function(hvf, x, nn)
    HNNLoss(hvf_function)
end

function (loss::HNNLoss)(   ::Union{Chain, AbstractExplicitLayer}, 
                            ps::Union{NeuralNetworkParameters, NamedTuple}, 
                            input::QPTOAT, 
                            output::QPTOAT)
    loss(ps, input, output)
end

function (loss::HNNLoss)(   ps::Union{NeuralNetworkParameters, NamedTuple},
                            input::QPTOAT,
                            output::QPTOAT)
    norm(loss.hvf(input, ps) - output) / norm(output)
end