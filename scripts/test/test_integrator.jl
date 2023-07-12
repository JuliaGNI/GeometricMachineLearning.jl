using GeometricMachineLearning
using GeometricEquations
using Test

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

hnn = NeuralNetwork(HamiltonianNeuralNetwork(1))
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

lnn = NeuralNetwork(LagrangianNeuralNetwork(1))
ics = (q = q₀, p = p₀, λ = λ₀)

prob_lnn = LNNProblem(lnn, tspan, Δt, ics)

@test typeof(prob_lnn) <: GeometricProblem
@test typeof(prob_lnn) <: LODEProblem
@test equtype(prob_lnn) == LODE

prob_lnn2 = LNNProblem(lnn, tspan, Δt, q₀, p₀)

#@test prob_lnn == prob_lnn2

#########################################
# Test for SympNetMethod Integrator
#########################################





