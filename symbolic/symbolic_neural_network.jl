using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

include("utils.jl")


abstract type SymbolicNeuralNetwork{AT, ET} <: AbstractNeuralNetwork end

neuralnet(snn::SymbolicNeuralNetwork) = snn.nn
architecture(snn::SymbolicNeuralNetwork) = snn.nn.architecture
model(snn::SymbolicNeuralNetwork) = snn.nn.model
params(snn::SymbolicNeuralNetwork) = snn.nn.params

(snn::SymbolicNeuralNetwork)(x, params = snn.params) = snn.eval(x, params)
apply(snn::SymbolicNeuralNetwork, x, args...) = snn(x, args...)



