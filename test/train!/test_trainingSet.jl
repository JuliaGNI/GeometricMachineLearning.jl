using GeometricMachineLearning
using Test

include("../macro_testerror.jl")
include("../data/data_generation.jl")

#########################################
# Test for TrainingParameters
#########################################

training_data = tra_ps_data

nruns = 10
method = ExactHnn()
mopt = GradientOptimizer()
bs = 10

training_parameters = TrainingParameters(nruns, method, mopt; batch_size = bs)

@test GeometricMachineLearning.nruns(training_parameters) == nruns
@test GeometricMachineLearning.method(training_parameters) == method
@test GeometricMachineLearning.opt(training_parameters) == mopt
@test GeometricMachineLearning.batchsize(training_parameters) == bs

#########################################
# Test for TrainingSet
#########################################

hnn = HamiltonianArchitecture(2; nhidden= 2, width = 5)
nn1 = NeuralNetwork(hnn, Float64)

training_set1 = TrainingSet(nn1, training_parameters, training_data)

@test GeometricMachineLearning.nn(training_set1) == nn1
@test parameters(training_set1) == training_parameters
@test data(training_set1) == training_data

lnn = LagrangianNeuralNetwork(2; nhidden= 2, width = 5)
nn2 = NeuralNetwork(hnn, Float64)

training_set2 = TrainingSet(nn2, training_parameters, training_data)

@test GeometricMachineLearning.nn(training_set2) == nn2
@test parameters(training_set1) == training_parameters
@test data(training_set1) == training_data

#########################################
# Test for EnsembleTraining
#########################################

ensemble_training = EnsembleTraining()

@test GeometricMachineLearning.size(ensemble_training) == 0
@test isnnShared(ensemble_training) == false
@test isParametersShared(ensemble_training) == false
@test isDataShared(ensemble_training) == false

@testerror GeometricMachineLearning.nn(ensemble_training) 
@testerror parameters(ensemble_training)
@testerror data(ensemble_training)

push!(ensemble_training, training_set1)

@test size(ensemble_training) == 1
@test isnnShared(ensemble_training) == true
@test isParametersShared(ensemble_training) == true
@test isDataShared(ensemble_training) == true

@testnoerror GeometricMachineLearning.nn(ensemble_training) 
@testnoerror parameters(ensemble_training)
@testnoerror data(ensemble_training)

@test GeometricMachineLearning.nn(ensemble_training) == nn1
@test parameters(ensemble_training) == training_parameters 
@test data(ensemble_training) == training_data

ensemble_training2 = EnsembleTraining(training_set2)
merge!(ensemble_training, ensemble_training2)

@test size(ensemble_training) == 2
@test isnnShared(ensemble_training) == false
@test isParametersShared(ensemble_training) == true
@test isDataShared(ensemble_training) == true

@testerror GeometricMachineLearning.nn(ensemble_training) 
@testnoerror parameters(ensemble_training)
@testnoerror data(ensemble_training)

@test parameters(ensemble_training) == training_parameters 
@test data(ensemble_training) == training_data