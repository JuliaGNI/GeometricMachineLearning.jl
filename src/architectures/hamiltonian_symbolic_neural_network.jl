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