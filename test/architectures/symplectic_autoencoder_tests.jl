using GeometricMachineLearning
using Test
using Zygote: jacobian
import Random

include("../helpers/poisson.jl")
include("../helpers/eltypes.jl")

Random.seed!(123)

function test_accuracy(::Type{T}, N::Integer, n::Integer; tol::Real = 0.35,
        n_epochs::Integer = 100) where {T}
    dl = DataLoader(rand(T, N, 10 * N); autoencoder = true)

    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n), T)

    o = TrainingOptimizer(sae_nn.params; algorithm = Adam())
    sae_error = train!(sae_nn, o, dl, Batch(10), n_epochs; show_progress = false)[end]

    @test eltype(sae_error) === T
    @test sae_error < tol
end

function test_encoder_and_decoder(::Type{T}, N::Integer, n::Integer) where {T}
    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n), T)
    sae_encoder = encoder(sae_nn)
    sae_decoder = decoder(sae_nn)

    test_vec = rand(T, N)
    test_mat = rand(T, N, N)

    @test eltype(sae_nn(test_vec)) === T
    @test sae_nn(test_vec) ≈ sae_decoder(sae_encoder(test_vec))
    @test sae_nn(test_mat) ≈ sae_decoder(sae_encoder(test_mat))
end

function test_symplecticity(::Type{T}, N::Integer, n::Integer) where {T}
    sae_nn = NeuralNetwork(SymplecticAutoencoder(N, n), T)
    sae_decoder = decoder(sae_nn)
    test_vector = rand(T, n)

    # this matrix should be symplectic
    sympl_mat = jacobian(vec -> sae_decoder(vec), test_vector)[1]
    @test eltype(sympl_mat) === T
    @test eltype(sympl_mat' * poisson_matrix(N, T) * sympl_mat) === T
    @test poisson_matrix(n, T) ≈ sympl_mat' * poisson_matrix(N, T) * sympl_mat

    # test if it's still symplectic after training
    dl = DataLoader(rand(T, N, 10 * N); autoencoder = true)
    o = TrainingOptimizer(sae_nn.params; algorithm = Adam())
    train!(sae_nn, o, dl, Batch(10), 10; show_progress = false)
    sympl_mat = jacobian(vec -> sae_decoder(vec), test_vector)[1]
    @test poisson_matrix(n, T) ≈ sympl_mat' * poisson_matrix(N, T) * sympl_mat
end

function test_decoder_output_layers(::Type{T}, N::Integer, n::Integer,
        n_output_layers::Integer) where {T}
    sae_nn = NeuralNetwork(
        SymplecticAutoencoder(N, n; n_decoder_output_layers = n_output_layers), T)
    sae_encoder = encoder(sae_nn)
    sae_decoder = decoder(sae_nn)

    # composition still works
    test_vec = rand(T, N)
    test_mat = rand(T, N, N)
    @test eltype(sae_nn(test_vec)) === T
    @test sae_nn(test_vec) ≈ sae_decoder(sae_encoder(test_vec))
    @test sae_nn(test_mat) ≈ sae_decoder(sae_encoder(test_mat))

    # decoder has exactly n_output_layers more layers than the baseline (n_output_layers = 0) decoder
    baseline_nn = NeuralNetwork(SymplecticAutoencoder(N, n; n_decoder_output_layers = 0), T)
    baseline_decoder = decoder(baseline_nn)
    @test length(sae_decoder.model.layers) ==
          length(baseline_decoder.model.layers) + n_output_layers

    # symplecticity still holds
    test_vector = rand(T, n)
    sympl_mat = jacobian(vec -> sae_decoder(vec), test_vector)[1]
    @test poisson_matrix(n, T) ≈ sympl_mat' * poisson_matrix(N, T) * sympl_mat
end

function all_tests(::Type{T}, N::Integer, n::Integer; kwargs...) where {T}
    test_accuracy(T, N, n; kwargs...)
    test_encoder_and_decoder(T, N, n; kwargs...)
    test_symplecticity(T, N, n)
end

for T in REAL_ELTYPES
    all_tests(T, 10, 6)
    all_tests(T, 20, 10)

    # test the nonlinear decoder output layers
    test_decoder_output_layers(T, 10, 6, 1)
    test_decoder_output_layers(T, 10, 6, 2)
    test_decoder_output_layers(T, 20, 10, 3)
end

@testset "block counts at the edges" begin
    # One block builds; a spacing that rounds to zero is refused
    @test NeuralNetwork(SymplecticAutoencoder(8, 4; n_encoder_blocks = 1), Float64) isa
          NeuralNetwork
    re = r"^(?:ArgumentError: )?n_blocks \(4\) must be at most 3 for full_dim \(8\) and reduced_dim \(4\)\.$"
    @test_throws re NeuralNetwork(SymplecticAutoencoder(8, 4; n_decoder_blocks = 4), Float64)
end
