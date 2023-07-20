using GeometricMachineLearning
using GeometricSolutions
using GeometricEquations
using Test

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: harmonic_oscillator_hode_ensemble, exact_solution, exact_solution_q, exact_solution_p

include("plots.jl")

#create the object ensemble_solution
ensemble_problem = harmonic_oscillator_hode_ensemble()
ensemble_solution =  EnsembleSolution(ensemble_problem)


for i in eachindex(ensemble_solution.s)
    ensemble_solution.s[i]= exact_solution(GeometricEquations.problem(ensemble_problem,i)) 
end


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
@test get_length_trajectory(training_data)[1]  == 10
@test get_nb_point(training_data)              === nothing

@test Tuple(keys(GeometricMachineLearning.get_data(training_data))) ==(:p, :q)

#creating a training sets
sympnet = NeuralNetwork(GSympNet(dim(training_data); nhidden = 3))
nruns = 10000
method = BasicSympNet()
mopt = MomentumOptimizer()
training_parameters = TrainingParameters(nruns, method, mopt)

training_set = TrainingSet(sympnet, training_parameters, training_data)

#training of the neural network
neural_net_solution = train!(training_set; showprogress = true)


q = []
p = []
qp = [0.2, 0.4]
for i in 1:1000
    global qp
    qp = neural_net_solution.nn(qp)
    push!(q,qp[1])
    push!(p,qp[2])
end

prediction = (q=q, p=p)

plots(training_data, prediction)


#integrate
#prediction = integrate(neural_net_solution)


















