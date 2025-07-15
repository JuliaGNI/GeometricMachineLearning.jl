"""
    HNNLoss <: NetworkLoss

The loss for a Hamiltonian neural network.

# Constructor

This can be called with a `NeuralNetwork`, built with a [`HamiltonianArchitecture`](@ref), as the only input arguemtn, i.e.:
```julia
HNNLoss(nn)
```
where `nn` is a `NeuralNetwork`, that is built with a [`HamiltonianArchitecture`](@ref), gives the corresponding Hamiltonian loss.

# Functor

```julia
loss(c, ps, input, output)
loss(ps, input, output) # equivalent to the above
```
"""
struct HNNLoss{FT} <: NetworkLoss
    hvf::FT
end

function HNNLoss(arch::HamiltonianArchitecture)
    HNNLoss(hamiltonian_vector_field(arch))
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