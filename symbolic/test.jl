using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

import GeometricMachineLearning: vectorfield

include("utils.jl")

include("symbolic_hnn.jl")



using Test

hnn = NeuralNetwork(HamiltonianNeuralNetwork(2), Float64)
shnn = Symbolize(hnn)

@test typeof(shnn) <: SymbolicHNN{<:HamiltonianNeuralNetwork}
@test architecture(shnn) == hnn.architecture
@test params(shnn) == hnn.params
@test model(shnn) == hnn.model

x = [0.5, 0.8]
@test shnn(x) == hnn(x)
@time shnn(x)
@time develop(params(shnn))
@time hnn(x)

#@test field(shnn,x) == vectorfield(hnn, x)
@time field(shnn, x)
@time vectorfield(hnn, x)


