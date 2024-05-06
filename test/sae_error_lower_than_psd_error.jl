using GeometricMachineLearning
using Test
import Random 

Random.seed!(123)

function test_accuracy(N::Integer, n::Integer; tol::Real = .35, n_epochs::Integer = 100)
    dl = DataLoader(rand(N, 10 * N); autoencoder = true)

    psd_nn = NeuralNetwork(PSDArch(N, n))
    psd_error = solve!(psd_nn, dl)

    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n; n_encoder_layers = 5, n_decoder_layers = 5))
    
    o = Optimizer(AdamOptimizer(), sae_nn)
    sae_error = o(sae_nn, dl, Batch(10), n_epochs)[end]

    @test sae_error < psd_error 
end

test_accuracy(10, 4)
test_accuracy(6, 2, n_epochs = 200)