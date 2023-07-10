using GeometricMachineLearning
using Test

include("generation_of_data.jl")
include("macro_testerror.jl")

#########################################
# Test NeuralNetSolution
#########################################

training_data = tra_ps_data
nn = NeuralNetwork(GSympNet(2))
mopt = GradientOptimizer()
method = BasicSympNet()
nruns = 0
batch_size = (1,2)

training_parameters = TrainingParameters(nruns, method, mopt; batch_size = batch_size)

neural_net_solution = train!(nn, training_data, training_parameters)

@test GeometricMachineLearning.nn(neural_net_solution)  == nn
@test problem(neural_net_solution)                      == UnknownProblem
@test tstep(neural_net_solution)                        == 0.1
@test length(loss(neural_net_solution))                 == nruns

@test size_history(neural_net_solution) == 0
@test nbtraining(neural_net_solution)   ==  1

@testnoerror set_sizemax_history(neural_net_solution, 7) = _set_sizemax_history(nns.history, sizemax)





#########################################
# Test SingleHistory
#########################################

#test last
@inline parameters(sh::SingleHistory) = sh.parameters
@inline datashape(sh::SingleHistory) = sh.datashape
@inline Base.size(sh::SingleHistory) = sh.size_data
@inline loss(sh::SingleHistory) = sh.loss


#########################################
# Test EnsembleNeuralNetSolution
#########################################