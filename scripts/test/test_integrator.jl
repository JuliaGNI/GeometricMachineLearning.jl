using GeometricMachineLearning
using GeometricEquations
using Test

t₀ = 0.
t₁ = 1.
q₀ = [0.5]
p₀ = [0.5]
Δt = 0.1
tspan = (t₀, t₁)

#########################################
# Test for HNNProblem
#########################################

hnn = NeuralNetwork(HamiltonianNeuralNetwork(1))
ics = (q = q₀, p = p₀)

prob = HNNProblem(hnn, tspan, Δt, ics)

@test typeof(prob) <: GeometricProblem
@test typeof(prob) <: HODEProblem
@test equtype(prob) == HODE

@test periodicity(prob).q == periodicity(equation(prob))
@test periodicity(prob).p == NullPeriodicity()

prob2 = HNNProblem(hnn, tspan, tstep, q₀, p₀)



#########################################
# Test for LNNProblem
#########################################



#########################################
# Test for SympNetMethod Integrator
#########################################





