using GeometricMachineLearning
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint
using Test
import Random

Random.seed!(123)

function test_reduced_vector_fields(reduced_dim::Integer)
    pr = hodeproblem()

    sol = integrate(pr, ImplicitMidpoint())

    dl = DataLoader(sol)

    model1 = PSDArch(dl.input_dim, reduced_dim)

    model2 = SymplecticAutoencoder(dl.input_dim, reduced_dim)

    nn1 = NeuralNetwork(model1)

    nn2 = NeuralNetwork(model2)

    rs1 = HRedSys(hodeproblem(), get_encoder(nn1), get_decoder(nn1))

    rs2 = HRedSys(hodeproblem(), get_encoder(nn2), get_decoder(nn2))

    # we expect the neural network projection & reduction errors to be higher than the PSD projection & reduction errors if we don't train
    @test compute_projection_error(rs1) < compute_projection_error(rs2)
    @test compute_reduction_error(rs1) < compute_reduction_error(rs2)
end

test_reduced_vector_fields(2)