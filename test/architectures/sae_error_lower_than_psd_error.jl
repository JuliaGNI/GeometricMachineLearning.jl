using GeometricMachineLearning
using GeometricMachineLearning: solve!
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(123)

function test_accuracy(::Type{T}, N::Integer, n::Integer; n_epochs::Integer = 100) where {T}
    # Seeded per call, for the reason given in `optimizers/svd_optim.jl`: this is called twice (and
    # once per precision), and a call inheriting its predecessor's RNG state made the comparison
    # below depend on how much randomness the optimizer consumes instead of on whether the
    # autoencoder beats PSD.
    Random.seed!(123)
    dl = DataLoader(rand(T, N, 10 * N); autoencoder = true)

    psd_nn = NeuralNetwork(PSDArch(N, n), T)
    psd_error = solve!(psd_nn, dl)

    sae_nn = NeuralNetwork(
        SymplecticAutoencoder(N, n; n_encoder_layers = 5, n_decoder_layers = 5),
        T)

    o = TrainingOptimizer(sae_nn.params; algorithm = Adam())
    sae_error = train!(sae_nn, o, dl, Batch(10), n_epochs; show_progress = false)[end]

    @test eltype(psd_error) === T
    @test eltype(sae_error) === T
    @test sae_error < psd_error
end

for T in REAL_ELTYPES
    test_accuracy(T, 10, 4)
    test_accuracy(T, 6, 2, n_epochs = 200)
end
