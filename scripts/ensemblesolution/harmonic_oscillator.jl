using GeometricMachineLearning
using GeometricSolutions
using GeometricEquations
using Test

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: harmonic_oscillator_hode_ensemble, exact_solution, exact_solution_q, exact_solution_p


#create the object ensemble_solution
ensemble_problem = harmonic_oscillator_hode_ensemble()
ensemble_solution =  EnsembleSolution(ensemble_problem)


for i in eachindex(ensemble_solution.s)
    ensemble_solution.s[i]= exact_solution(GeometricEquations.problem(ensemble_problem,i)) 
end


#create the data associated
training_data = TrainingData(ensemble_solution)

@test GeometricMachineLearning.problem(training_data)               == ensemble_problem
@test typeof(shape(training_data))         == TrajectoryData
@test type(data_symbols(training_data))    == PhaseSpaceSymbol
@test symbols(training_data)               == (:q,:p)
@test dim(training_data)                   == 2
@test noisemaker(training_data)            == NothingFunction()    

#@test get_Δt(training_data)                == 0.1
@test get_nb_trajectory(training_data)     == 100
#@test get_length_trajectory(training_data) == 11
@test get_nb_point(training_data)         === nothing

@test Tuple(keys(GeometricMachineLearning.get_data(training_data))) ==(:p, :q)

#creating a training sets
sympnet = NeuralNetwork(GSympNet(dim(training_data); nhidden = 3))
nruns = 10
method = BasicSympNet()
mopt = MomentumOptimizer()
training_parameters = TrainingParameters(nruns, method, mopt)

training_set = TrainingSet(sympnet, training_parameters, training_data)

#training of the neural network
neural_net_solution = train!(training_set; showprogress = true)


include("../plots.jl")

H(x) = hamiltonian([x[2]],0.0, [x[1]],(k =0.5, ω = sqrt(0.5)))



#plot_hnn(H, sympnet, loss(neural_net_solution); filename="harmonic_oscillator_ensemble.png", xmin=-1.2, xmax=+1.2, ymin=-1.2, ymax=+1.2)











