using GeometricMachineLearning
using Test

include("generation_of_data.jl")
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

neural_net_solution = train!(nn, training_data, training_parameters)

#########################################
# Test train! with TrainingSet
#########################################

training_set1 = TrainingSet(nn, training_parameters, training_data)
training_set2 = TrainingSet(nn, training_parameters, training_data)

ensemble_nns = train!(training_set1, training_set2)

#########################################
# Test train! with EnsembleTraining
#########################################

ensemble_training = EnsembleTraining(training_set1, training_set2)

ensemble_nns = train!(ensemble_training)


#########################################
# Test train! with NeuralNetSolution
#########################################

train!(nns::NeuralNetSolution, data::AbstractTrainingData, tp::TrainingParameters)

train!(nns::NeuralNetSolution, ts::TrainingSet)


