using GeometricMachineLearning
using Test
import Random 

Random.seed!(123)

function test_accuracy(N::Integer, n::Integer; n_epochs::Integer = 100)
    # Seeded per call, for the reason given in
    # `optimizers/optimizer_convergence_tests/svd_optim.jl`: this is called twice, and the second
    # call inheriting the first call's RNG state made the comparison below depend on how much
    # randomness the optimizer consumes instead of on whether the autoencoder beats PSD.
    Random.seed!(123)
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)

    psd_nn = NeuralNetwork(PSDArch(N, n))
    psd_error = solve!(psd_nn, dl)

    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n; n_encoder_layers = 5, n_decoder_layers = 5))
    
    o = Optimizer(Adam(), sae_nn)
    sae_error = o(sae_nn, dl, Batch(10), n_epochs; show_progress = false)[end]

    @test sae_error < psd_error 
end

test_accuracy(10, 4)
test_accuracy(6, 2, n_epochs = 200)