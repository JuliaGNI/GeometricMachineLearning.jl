using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

include("utils.jl")
include("symbolic.jl")



abstract type SymbolicNeuralNetwork{AT, ET} <: AbstractNeuralNetwork end

neuralnet(snn::SymbolicNeuralNetwork) = snn.nn
architecture(snn::SymbolicNeuralNetwork) = snn.nn.architecture
model(snn::SymbolicNeuralNetwork) = snn.nn.model
params(snn::SymbolicNeuralNetwork) = snn.nn.params

(snn::SymbolicNeuralNetwork)(x, params = params(shnn)) = snn.est(x, develop(params)...)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)