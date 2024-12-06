"""
    SymbolicPullback(arch::HamiltonianArchitecture)

Make a `SymbolicPullback` based on a [`HamiltonianArchitecture`](@ref).

# Implementation

Internally this is calling `SymbolicNeuralNetwork` and [`HnnLoss`](@ref).
"""
function SymbolicPullback(arch::HamiltonianArchitecture)
    nn = SymbolicNeuralNetwork(arch)
    loss = HNNLoss(arch)
    SymbolicPullback(nn, loss)
end