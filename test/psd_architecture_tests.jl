using GeometricMachineLearning
using Zygote: jacobian
using Test
import Random 

Random.seed!(123)

function test_accuracy(N::Integer, n::Integer; tol::Real = .35)
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)

    psd_nn = NeuralNetwork(PSDArch(N, n))
    psd_error = solve!(psd_nn, dl)

    @test psd_error < tol 
end

function test_encoder_and_decoder(N::Integer, n::Integer)
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)

    psd_nn = NeuralNetwork(PSDArch(N, n))
    psd_encoder = get_encoder(psd_nn)
    psd_decoder = get_decoder(psd_nn)

    test_vec = rand(N)
    test_mat = rand(N, N)

    @test psd_nn(test_vec) ≈ psd_decoder(psd_encoder(test_vec))
    @test psd_nn(test_mat) ≈ psd_decoder(psd_encoder(test_mat))
end

function test_symplecticity(N::Integer, n::Integer)
    psd_nn = NeuralNetwork(PSDArch(N, n))
    psd_decoder = get_decoder(psd_nn)
    test_vector = rand(n)

    # this matrix should be symplectic
    sympl_mat = jacobian(vec -> psd_decoder(vec), test_vector)[1]
    @test SymplecticPotential(n) ≈ sympl_mat' * SymplecticPotential(N) * sympl_mat

    # test if it's still symplectic after computing psd 
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)
    solve!(psd_nn, dl)
    sympl_mat = jacobian(vec -> psd_decoder(vec), test_vector)[1]
    @test SymplecticPotential(n) ≈ sympl_mat' * SymplecticPotential(N) * sympl_mat
end

function all_tests(N::Integer, n::Integer; kwargs...)
    test_accuracy(N, n; kwargs...)
    test_encoder_and_decoder(N, n; kwargs...)
    test_symplecticity(N, n)
end

all_tests(10, 6)
all_tests(20, 10)