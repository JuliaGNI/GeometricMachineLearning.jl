using GeometricMachineLearning
using GeometricSolutions
using Test

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: harmonic_oscillator_hode_ensemble


#create the object ensemble_solution
ensemble_solution = harmonic_oscillator_hode_ensemble()

#create the data associated
data_training = TrainingData(ensemble_solution)

@test problem(training_data)               == UnknownProblem()
@test typeof(shape(training_data))         == TrajectoryData
@test type(data_symbols(training_data))    == PhaseSpaceSymbol
@test symbols(training_data)               == (:q,:p)
@test dim(training_data)                   == 2
@test noisemaker(training_data)            == NothingFunction()    

@test get_Δt(training_data)                == 0.1
@test get_nb_trajectory(training_data)     == 10
@test get_length_trajectory(training_data) == 11
@test get_nb_point(training_data)         === nothing

@test Tuple(keys(GeometricMachineLearning.get(training_data))) ==(:q, :p)

#creating a training sets
sympnet = GSympNet(dim(data_training); nhidden = 3)
nruns = 1000
method = BasicSympNet()
mopt = MomentumOptimizer()
training_paramters = TrainingParameters(nruns, method, mopt)

training_set = TrainingSet(sympnet, training_parameters, training_data)

#training of the neural network
neural_net_solution = train!(training_set)












