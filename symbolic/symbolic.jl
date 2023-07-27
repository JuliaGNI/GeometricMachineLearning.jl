using GeometricMachineLearning
using AbstractNeuralNetworks

import AbstractNeuralNetworks: NeuralNetwork

using Symbolics



symbolicParams(nn::NeuralNetwork) = symbolicParams(nn.params)

symbolicParams(M::AbstractArray) = (@variables M[Tuple([1:s for s in size(M)])...])[1]

symbolicParams(nt::NamedTuple) = NamedTuple{keys(nt)}(symbolicParams(M) for M in values(nt))

symbolicParams(t::Tuple) = Tuple([symbolicParams(e) for e in t])








sympnet = GSympNet(4)
nn = NeuralNetwork(sympnet, Float64)

@variables sinput[1:dim(nn.architecture)]

sparams = symbolicParams(nn)

nn(sinput, sparams)









