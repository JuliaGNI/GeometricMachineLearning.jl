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
nruns = 1
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

#########################################
# Test train! for all training methods
#########################################

hnn = NeuralNetwork(HamiltonianNeuralNetwork(2), Float64)
exacthnn = ExactHnn()
sympeuler = SEuler()

#@testnoerror  train!(hnn, sam_dps_data, mopt, exacthnn; ntraining = nruns)
#@testnoerror  train!(hnn, tra_ps_data, mopt, sympeuler; ntraining = nruns)

lnn = NeuralNetwork(LagrangianNeuralNetwork(2), Float64)
exactlnn = ExactLnn()
midpointlnn = VariaMidPoint()

#@testnoerror  train!(lnn, sam_accposvel_data, mopt, exactlnn; ntraining = nruns)
#@testerror  train!(lnn, tra_pos_data, mopt, exactlnn; ntraining = nruns)