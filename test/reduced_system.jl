using GeometricMachineLearning
using GeometricProblems.CoupledHarmonicOscillator: hodeproblem
using GeometricIntegrators: integrate, ImplicitMidpoint, ExplicitEulerRK, ExplicitMidpoint
using FiniteDifferences: jacobian, central_fdm
using Test
import Random

Random.seed!(123)

function set_up_reduced_systems(reduced_dim::Integer, integrator)
    pr = hodeproblem()

    sol = integrate(pr, integrator)

    dl = DataLoader(sol)

    model1 = PSDArch(dl.input_dim, reduced_dim)

    # Here the number of decoder blocks is set manually because the default is too big! 
    model2 = SymplecticAutoencoder(dl.input_dim, reduced_dim; activation = x -> log(1. + exp(x)), n_encoder_layers = 20, n_decoder_blocks = 2)

    nn1 = NeuralNetwork(model1)

    nn2 = NeuralNetwork(model2)

    rs1 = HRedSys(hodeproblem(), encoder(nn1), decoder(nn1); integrator = integrator)

    rs2 = HRedSys(hodeproblem(), encoder(nn2), decoder(nn2); integrator = integrator)

    rs1, rs2 
end

function test_reduced_vector_fields(reduced_dim::Integer, integrator)
    rs1, rs2 = set_up_reduced_systems(reduced_dim, integrator)

    # we expect the neural network projection & reduction errors to be higher than the PSD projection & reduction errors if we don't train
    @test projection_error(rs1) < projection_error(rs2)
    @test reduction_error(rs1) < reduction_error(rs2)
end

function test_if_reduced_vector_fields_are_divergence_free(v_reduced::Function, f_reduced::Function, parameters::NamedTuple; tol = 1e-10)
    function v_reduced_explicit(q, p)
        v = zero(q)
        v_reduced(v, 0, q, p, parameters)
        v
    end
    function f_reduced_explicit(q, p)
        f = zero(p)
        f_reduced(f, 0, q, p, parameters)
        f
    end

    q̃ = rand(1)
    p̃ = rand(1)
    compute_derivative_q(f) = sum(jacobian(central_fdm(10, 1), q -> f(q, p̃), q̃)[1])
    compute_derivative_p(f) = sum(jacobian(central_fdm(10, 1), p -> f(q̃, p), p̃)[1])
    div_estimate = compute_derivative_q(v_reduced_explicit) + compute_derivative_p(f_reduced_explicit)
    @test abs(div_estimate) < tol
end

function test_if_reduced_vector_fields_are_divergence_free(rs::HRedSys)
    test_if_reduced_vector_fields_are_divergence_free(rs.v_reduced, rs.f_reduced, rs.parameters)

    nothing
end

function check_if_reduced_vector_fields_are_divergence_free(reduced_dim::Integer, integrator)
    rs1, rs2 = set_up_reduced_systems(reduced_dim, integrator)

    test_if_reduced_vector_fields_are_divergence_free(rs1)
    test_if_reduced_vector_fields_are_divergence_free(rs2)

    nothing
end

test_reduced_vector_fields(2, ImplicitMidpoint())
test_reduced_vector_fields(2, ExplicitMidpoint())

check_if_reduced_vector_fields_are_divergence_free(2, ImplicitMidpoint())
check_if_reduced_vector_fields_are_divergence_free(2, ExplicitMidpoint())
check_if_reduced_vector_fields_are_divergence_free(2, ExplicitEulerRK())