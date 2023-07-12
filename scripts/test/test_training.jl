using GeometricMachineLearning
using Test

include("data_generation.jl")
include("macro_testerror.jl")

#########################################
# Test the basic train! function
#########################################

training_data = tra_ps_data
nn = NeuralNetwork(GSympNet(2))
mopt = GradientOptimizer()
method = BasicSympNet()
nruns = 0
batch_size = (1,2)

@testnoerror total_loss = train!(nn, training_data, mopt, method; ntraining = nruns, batch_size = batch_size)

@test typeof(total_loss) <: AbstractArray
@test length(total_loss) == nruns

#########################################
# Test train! with TrainingParameters
#########################################

training_parameters = TrainingParameters(nruns, method, mopt; batch_size = batch_size)

@testnoerror neural_net_solution = train!(nn, training_data, training_parameters)

#########################################
# Test train! with TrainingSet
#########################################

training_set1 = TrainingSet(nn, training_parameters, training_data)

@testnoerror ensemble_nns = train!(training_set1)

#########################################
# Test train! with EnsembleTraining
#########################################

ensemble_training = EnsembleTraining(training_set1, training_set1)

@testnoerror ensemble_nns = train!(ensemble_training)

#########################################
# Test train! with NeuralNetSolution
#########################################

@testnoerror train!(neural_net_solution, training_data, training_parameters)

@testnoerror train!(neural_net_solution, training_set1)

