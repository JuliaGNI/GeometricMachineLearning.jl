using GeometricMachineLearning
using GeometricEquations
using Test

include("data_generation.jl")
include("macro_testerror.jl")

t₀ = 0.
t₁ = 1.
q₀ = [0.5]
p₀ = [0.5]
v₀ = [0.5]
λ₀ = [0.0]
Δt = 0.1
tspan = (t₀, t₁)

#########################################
# Test for HNNProblem
#########################################

hnn = NeuralNetwork(HamiltonianNeuralNetwork(2))
ics = (q = q₀, p = p₀)

prob_hnn = HNNProblem(hnn, tspan, Δt, ics)

@test typeof(prob_hnn) <: GeometricProblem
@test typeof(prob_hnn) <: HODEProblem
@test equtype(prob_hnn) == HODE

prob_hnn2 = HNNProblem(hnn, tspan, Δt, q₀, p₀)

@test prob_hnn == prob_hnn2

#########################################
# Test for LNNProblem
#########################################

lnn = NeuralNetwork(LagrangianNeuralNetwork(2))
ics = (q = q₀, p = p₀, λ = λ₀)

prob_lnn = LNNProblem(lnn, tspan, Δt, ics)

@test typeof(prob_lnn) <: GeometricProblem
@test typeof(prob_lnn) <: LODEProblem
@test equtype(prob_lnn) == LODE

prob_lnn2 = LNNProblem(lnn, tspan, Δt, q₀, p₀, λ₀)

#@test prob_lnn == prob_lnn2

#########################################
# Test for SympNetMethod Integrator
#########################################

training_data = TrainingData(Data_traps , get_Data_traps, prob_hnn)
neuralnet = NeuralNetwork(GSympNet(2))
mopt = GradientOptimizer()
method = BasicSympNet()
nruns = 0
batch_size = (1,2)
training_parameters = TrainingParameters(nruns, method, mopt; batch_size = batch_size)

neural_net_solution = train!(neuralnet, training_data, training_parameters)

#@testnoerror sol = integrate(neural_net_solution)





