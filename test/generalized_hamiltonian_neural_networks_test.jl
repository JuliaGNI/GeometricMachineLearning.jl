using GeometricMachineLearning
using GeometricMachineLearning: OptionalParameters, OneInitializer
using GeometricProblems.HarmonicOscillator: odeproblem, default_parameters
using GeometricIntegrators
using Test

sol = integrate(odeproblem(), ImplicitMidpoint())
dim = length(sol.problem.ics.q)

dl = DataLoader(sol)

function test_ghnn_without_parameters(dim::Integer = dim)
    arch = GeneralizedHamiltonianArchitecture(dim)
    nn = NeuralNetwork(arch; initializer=OneInitializer())
    @test nn([1., 1.]) ≈ [1.003217200759985, 0.9968055760434815]
end

function test_ghnn_with_parameters(dim::Integer = dim, parameters::OptionalParameters = default_parameters)
    arch = GeneralizedHamiltonianArchitecture(dim, parameters = parameters)
    nn = NeuralNetwork(arch; initializer=OneInitializer())
    @test nn([1., 1.], parameters) ≈ [1.0000350420844089, 0.9999649603746436]
end

test_ghnn_without_parameters()
test_ghnn_with_parameters()