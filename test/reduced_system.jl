using GeometricMachineLearning
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint, ExplicitEulerRK
using Test
import Random

Random.seed!(123)

function test_reduced_vector_fields(reduced_dim::Integer, integrator)
    pr = hodeproblem()

    sol = integrate(pr, integrator)

    dl = DataLoader(sol)

    model1 = PSDArch(dl.input_dim, reduced_dim)

    # Here the number of decoder blocks is set manually because the default is too big! 
    model2 = SymplecticAutoencoder(dl.input_dim, reduced_dim; n_decoder_blocks = 2)

    nn1 = NeuralNetwork(model1)

    nn2 = NeuralNetwork(model2)

    rs1 = HRedSys(hodeproblem(), get_encoder(nn1), get_decoder(nn1); integrator = integrator)

    rs2 = HRedSys(hodeproblem(), get_encoder(nn2), get_decoder(nn2); integrator = integrator)

    # we expect the neural network projection & reduction errors to be higher than the PSD projection & reduction errors if we don't train
    @test projection_error(rs1) < projection_error(rs2)
    @test reduction_error(rs1) < reduction_error(rs2)
end

# test_reduced_vector_fields(2, ImplicitMidpoint())
test_reduced_vector_fields(2, ExplicitEulerRK())