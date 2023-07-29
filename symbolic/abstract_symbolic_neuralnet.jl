using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

include("utils.jl")


abstract type AbstractSymbolicNeuralNetwork{AT, ET} <: AbstractNeuralNetwork end

neuralnet(snn::AbstractSymbolicNeuralNetwork) = snn.nn
architecture(snn::AbstractSymbolicNeuralNetwork) = snn.nn.architecture
model(snn::AbstractSymbolicNeuralNetwork) = snn.nn.model
params(snn::AbstractSymbolicNeuralNetwork) = snn.nn.params

(snn::AbstractSymbolicNeuralNetwork)(x, params = params(shnn)) = snn.est(x, develop(params)...)
apply(snn::AbstractSymbolicNeuralNetwork, x, args...) = snn(x, args...)
