using GeometricMachineLearning
using Test
using Zygote: jacobian
import Random 

Random.seed!(123)

function test_accuracy(N::Integer, n::Integer; tol::Real = .35, n_epochs::Integer = 100)
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)

    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n))
    
    o = Optimizer(AdamOptimizer(), sae_nn)
    sae_error = o(sae_nn, dl, Batch(10), n_epochs)[end]

    @test sae_error < tol 
end

function test_encoder_and_decoder(N::Integer, n::Integer)

    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n))
    sae_encoder = encoder(sae_nn)
    sae_decoder = decoder(sae_nn)

    test_vec = rand(N)
    test_mat = rand(N, N)

    @test sae_nn(test_vec) ≈ sae_decoder(sae_encoder(test_vec))
    @test sae_nn(test_mat) ≈ sae_decoder(sae_encoder(test_mat))
end

function test_symplecticity(N::Integer, n::Integer)
    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n))
    sae_decoder = decoder(sae_nn)
    test_vector = rand(n)
    
    # this matrix should be symplectic
    sympl_mat = jacobian(vec -> sae_decoder(vec), test_vector)[1]
    @test SymplecticPotential(n) ≈ sympl_mat' * SymplecticPotential(N) * sympl_mat

    # test if it's still symplectic after training
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)
    o = Optimizer(AdamOptimizer(), sae_nn)
    o(sae_nn, dl, Batch(10), 10)
    sympl_mat = jacobian(vec -> sae_decoder(vec), test_vector)[1]
    @test SymplecticPotential(n) ≈ sympl_mat' * SymplecticPotential(N) * sympl_mat
end

function all_tests(N::Integer, n::Integer; kwargs...)
    test_accuracy(N, n; kwargs...)
    test_encoder_and_decoder(N, n; kwargs...)
    test_symplecticity(N, n)
end

all_tests(10, 6)
all_tests(20, 10)