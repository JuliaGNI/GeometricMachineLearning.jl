using GeometricMachineLearning
using Test

include("data_generation.jl")
include("macro_testerror.jl")

#########################################
# Test of NeuralNetSolution and History
#########################################

training_data = tra_ps_data
neuralnet = NeuralNetwork(GSympNet(2))
mopt = GradientOptimizer()
method = SEuler()
nruns = 0
batch_size = (1,2)

training_parameters = TrainingParameters(nruns, method, mopt; batch_size = batch_size)

neural_net_solution = train!(neuralnet, training_data, training_parameters)

@test nn(neural_net_solution)            == neuralnet
@test problem(neural_net_solution)       == UnknownProblem()
@test tstep(neural_net_solution)         == 0.1
@test length(loss(neural_net_solution))  == nruns
@test size_history(neural_net_solution)  == 0
@test nbtraining(neural_net_solution)    ==  1

set_sizemax_history(neural_net_solution, 7)

last_sg1  = last(neural_net_solution)

@test parameters(last_sg1)   == training_parameters
@test datashape(last_sg1)    == shape(training_data)
@test size(last_sg1)         == size(training_data)
@test loss(last_sg1)         == loss(neural_net_solution)


training_parameters2 = TrainingParameters(2*nruns, ExactHnn(), AdamOptimizer())
training_data2 = sam_dps_data
train!(neural_net_solution, training_data2, training_parameters2)

@test nn(neural_net_solution)            == neuralnet
@test problem(neural_net_solution)       == UnknownProblem()
@test tstep(neural_net_solution)         == 0.1
@test length(loss(neural_net_solution))  == 2*nruns
@test size_history(neural_net_solution)  == 1
@test nbtraining(neural_net_solution)    ==  2

@test history(neural_net_solution)[1]    == last_sg1

last_sg2  = last(neural_net_solution)

@test parameters(last_sg2)   == training_parameters2
@test datashape(last_sg2)    == shape(training_data2)
@test size(last_sg2)         == size(training_data2)
@test loss(last_sg2)         == loss(neural_net_solution)

training_parameters3 = TrainingParameters(nruns, ExactHnn(), MomentumOptimizer())
training_data3 = sam_dps_data
train!(neural_net_solution, training_data3, training_parameters3)

@test nn(neural_net_solution)            == neuralnet
@test problem(neural_net_solution)       == UnknownProblem()
@test tstep(neural_net_solution)         == 0.1
@test length(loss(neural_net_solution))  == nruns
@test size_history(neural_net_solution)  == 2
@test nbtraining(neural_net_solution)    ==  3

@test history(neural_net_solution)[1]    == last_sg2
@test history(neural_net_solution)[2]    == last_sg1

last_sg3  = last(neural_net_solution)

@test parameters(last_sg3)   == training_parameters3
@test datashape(last_sg3)    == shape(training_data3)
@test size(last_sg3)         == size(training_data3)
@test loss(last_sg3)         == loss(neural_net_solution)

@testnoerror set_sizemax_history(neural_net_solution, 1)

@test size_history(neural_net_solution)  == 1
@test nbtraining(neural_net_solution)    ==  3

@test last(neural_net_solution)        == last_sg3
@test history(neural_net_solution)[1]  == last_sg2
@testerror history(neural_net_solution)[2]


#########################################
# Test EnsembleNeuralNetSolution
#########################################

training_set1 = TrainingSet(neuralnet, training_parameters, training_data)

neuralnet2 = NeuralNetwork(GSympNet(2; nhidden = 4))
training_set2 = TrainingSet(neuralnet2, training_parameters, training_data)

ensemble_training = EnsembleTraining(training_set1, training_set2)

ensemble_nnsolution = train!(ensemble_training)

@test size(ensemble_nnsolution) == 2

push!(ensemble_nnsolution, neural_net_solution)

@test size(ensemble_nnsolution) == 3

merge!(ensemble_nnsolution, ensemble_nnsolution)

@test size(ensemble_nnsolution) == 6

