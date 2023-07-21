using GeometricMachineLearning
using GeometricSolutions
using GeometricEquations
using GeometricEquations
using Test

using GeometricProblems.HarmonicOscillator
using GeometricProblems.HarmonicOscillator: harmonic_oscillator_hode_ensemble, hamiltonian# exact_solution, exact_solution_q, exact_solution_p


sgn(x) = x>=0 ? 1 : -1

function exact_solution(prob::Union{PODEProblem,HODEProblem})
    sol = GeometricSolution(prob)
    for n in eachtimestep(sol)
        sol.q[n] = [exact_solution_q(sol.t[n], sol.q[0], sol.p[0], GeometricEquations.parameters(prob))]
        sol.p[n] = [exact_solution_p(sol.t[n], sol.q[0], sol.p[0], GeometricEquations.parameters(prob))]
    end
    return sol
end


A(t, q, p, params) = sqrt(q^2 + p^2 / params.k) * sgn(-p)
ϕ(t, q, p, params) = acos(q / A(t, q, p, params))

exact_solution_q(t, q, p, params) = A(t, q, p, params) * cos(params.ω * t + ϕ(t, q, p, params))
exact_solution_p(t, q, p, params) = - params.ω * A(t, q, p, params) * sin(params.ω * t + ϕ(t, q, p, params))

exact_solution_q(t, q::AbstractVector, p::AbstractVector, params) = exact_solution_q(t, q[1], p[1], params)
exact_solution_p(t, q::AbstractVector, p::AbstractVector, params) = exact_solution_p(t, q[1], p[1], params)

exact_solution_q(t, x::AbstractVector, params) = exact_solution_q(t, x[1], x[2], params)
exact_solution_p(t, x::AbstractVector, params) = exact_solution_p(t, x[1], x[2], params)
exact_solution(t, x::AbstractVector, params) = [exact_solution_q(t, x, params), exact_solution_p(t, x, params)]

A(t, q, p, params) = sqrt(q^2 + p^2 / params.k) * sgn(-p)
ϕ(t, q, p, params) = acos(q / A(t, q, p, params))

exact_solution_q(t, q, p, params) = A(t, q, p, params) * cos(params.ω * t + ϕ(t, q, p, params))
exact_solution_p(t, q, p, params) = - params.ω * A(t, q, p, params) * sin(params.ω * t + ϕ(t, q, p, params))

exact_solution_q(t, q::AbstractVector, p::AbstractVector, params) = exact_solution_q(t, q[1], p[1], params)
exact_solution_p(t, q::AbstractVector, p::AbstractVector, params) = exact_solution_p(t, q[1], p[1], params)

exact_solution_q(t, x::AbstractVector, params) = exact_solution_q(t, x[1], x[2], params)
exact_solution_p(t, x::AbstractVector, params) = exact_solution_p(t, x[1], x[2], params)
exact_solution(t, x::AbstractVector, params) = [exact_solution_q(t, x, params), exact_solution_p(t, x, params)]

#create the object ensemble_solution
ensemble_problem = harmonic_oscillator_hode_ensemble()
ensemble_solution =  EnsembleSolution(ensemble_problem)

include("plots.jl")

#create the object ensemble_solution
ensemble_problem = harmonic_oscillator_hode_ensemble(tspan = (0.0,4.0))
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
@test get_length_trajectory(training_data)[1]  == 40
@test get_nb_point(training_data)              === nothing

@test Tuple(keys(GeometricMachineLearning.get_data(training_data))) ==(:p, :q)

#creating a training sets
sympnet = NeuralNetwork(GSympNet(dim(training_data); nhidden = 4, width = 10))
nruns = 10000
method = BasicSympNet()
mopt = AdamOptimizer()
training_parameters = TrainingParameters(nruns, method, mopt)

training_set = TrainingSet(sympnet, training_parameters, training_data)

#training of the neural network
neural_net_solution = train!(training_set; showprogress = true)


plot_result(training_data, neural_net_solution, hamiltonian; batch_nb_trajectory = 10, filename = "GSympNet 4-10 on Harmonic Oscillator")

#integrate
#prediction = integrate(neural_net_solution)






