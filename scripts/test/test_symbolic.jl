using SymbolicNeuralNetworks
using GeometricMachineLearning
using Test

include("data_generation.jl")
include("macro_testerror.jl")

########################################################################
# Test Symbolize a NeuralNetwork
########################################################################

training_data = tra_ps_data
nn = NeuralNetwork(HamiltonianArchitecture(2; nhidden=2), Float64)
mopt = GradientOptimizer()
method = SEulerA()
nruns = 1
batch_size = (1,2,2)
index_batch = get_batch(training_data, batch_size; check = false)

@testnoerror snn = Symbolize(nn, 2)

@test typeof(snn) <: SymbolicNeuralNetwork{<:HamiltonianArchitecture}

@test neuralnet(snn)    == nn
#@test architecture(snn) == nn.architecture 
@test params(snn)       == nn.params
@test model(snn)        == nn.model

x = [1,2]
@test snn(x) == nn(x)

#=
using Zygote

(::Zygote.ProjectTo{Float64})(x::Tuple{Float64}) = only(x)

(::Zygote.ProjectTo{AbstractArray})(x::Tuple{Vararg{Any}})  = [x...]

Base.size(nt::NamedTuple) = (length(nt),)

Zygote.gradient(x->sum(snn(x)), x)

GeometricMachineLearning.loss_gradient(snn, method, training_data, index_batch)
=#

########################################################################
# Test train a SymbolicNeuralNetwork
########################################################################
Base.size(nt::NamedTuple) = (length(nt),)

total_loss = train!(snn, training_data, mopt, method; ntraining = nruns, batch_size = batch_size)

@test typeof(total_loss) <: AbstractArray
@test length(total_loss) == nruns

########################################################################
# Copare performace between NeuralNetwork and SymbolicNeuralNetwork
########################################################################

@time train!(nn, training_data, mopt, method; ntraining = nruns, batch_size = batch_size)
@time train!(snn, training_data, mopt, method; ntraining = nruns, batch_size = batch_size)
