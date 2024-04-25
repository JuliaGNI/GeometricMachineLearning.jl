using GeometricMachineLearning
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint
using Test
import Random 

Random.seed!(123)

function test_reduced_vector_fields(full_dim::Integer, reduced_dim::Integer, time_steps::Integer=100, n_parameters::Integer=5)
    pr = hodeproblem()

    sol = integrate(pr, ImplicitMidpoint())

    dl = DataLoader(sol)

    model1 = PSDArch(full_dim, reduced_dim)

    model2 = SymplecticAutoencoder(full_dim, reduced_dim)

    nn1 = NeuralNetwork(model1)

    nn2 = NeuralNetwork(model2)

    rs1 = ReducedSystem(hodeproblem(); encoder = get_encoder(nn1), decoder = get_decoder(nn1))

    rs2 = ReducedSystem(hodeproblem(); encoder = get_encoder(nn2), decoder = get_decoder(nn2))

    # we expect the neural network reduction error to be higher than the PSD reduction error if we don't train
    @test compute_reduction_error(rs1) < compute_reduction_error(rs2)
end

test_reduced_vector_fields(20, 10)
