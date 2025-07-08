using GeometricMachineLearning
using GeometricSolutions
using GeometricEquations
using Test

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: hodeensemble, hamiltonian, default_parameters


#create the object ensemble_solution
ensemble_problem = hodeensemble(timespan = (0.0,4.0))
ensemble_solution =  exact_solution(ensemble_problem ) 

include("plots.jl")


#create the data associated
training_data = TrainingData(ensemble_solution)

@test GeometricMachineLearning.problem(training_data)   == ensemble_problem
@test typeof(shape(training_data))         == TrajectoryData
@test type(data_symbols(training_data))    == PhaseSpaceSymbol
@test symbols(training_data)               == (:q,:p)
@test dim(training_data)                   == 2
@test noisemaker(training_data)            == NothingFunction()    

@test get_Δt(training_data)                    == 0.1
@test get_nb_trajectory(training_data)         == 100
@test get_length_trajectory(training_data)[1]  == 40
@test get_nb_point(training_data)              === nothing

@test Tuple(keys(GeometricMachineLearning.get_data(training_data))) ==(:p, :q)

#creating a training sets
sympnet = NeuralNetwork(GSympNet(dim(training_data); nhidden = 4, width = 10), Float64)
nruns = 10000
method = BasicSympNet()
mopt = AdamOptimizer()
training_parameters = TrainingParameters(nruns, method, mopt)

training_set = TrainingSet(sympnet, training_parameters, training_data)

#training of the neural network
neural_net_solution = train!(training_set; showprogress = true)


H(x) = hamiltonian(x[1+length(x)÷2:end], 0.0, x[1:length(x)], default_parameters)
plot_result(training_data, neural_net_solution, H; batch_nb_trajectory = 10, filename = "GSympNet 4-10 on Harmonic Oscillator", nb_prediction = 5)







