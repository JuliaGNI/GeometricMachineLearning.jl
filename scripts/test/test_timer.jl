using GeometricMachineLearning
using Test

include("data_generation.jl")
include("macro_testerror.jl")

#########################################
# Test the basic train! function
#########################################

training_data = tra_ps_data
nn = NeuralNetwork(GSympNet(2; nhidden=2), Float64)
mopt = GradientOptimizer()
method = BasicSympNet()
nruns = 1000

@test_nowarn train!(nn, training_data, mopt, method; ntraining = nruns, timer = true)