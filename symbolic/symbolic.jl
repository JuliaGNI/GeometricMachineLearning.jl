using GeometricMachineLearning
using LinearAlgebra
using Symbolics

using AbstractNeuralNetworks
import AbstractNeuralNetworks: NeuralNetwork

include("utils.jl")
include("symbolic_params.jl")


function symbolicModel(nn::NeuralNetwork)
    @variables sinput[1:dim(nn.architecture)]
    sparams =  symbolicParams(nn)
    nn(sinput, sparams)
end


function symbolicField(nn::NeuralNetwork{<:HamiltonianNeuralNetwork})


end

