using Test
using GeometricMachineLearning
using GeometricMachineLearning: solve!
using LinearAlgebra: norm
using Zygote: Zygote
import Random

@testset "Loss docstring examples" begin
    Random.seed!(123)
    d = 2
    seq_length = 3
    prediction_window = 2
    nn = NeuralNetwork(StandardTransformerIntegrator(d))
    input_mat = [1.0 2.0 3.0; 4.0 5.0 6.0]
    output_mat = [1.0 2.0; 3.0 4.0]
    loss = TransformerLoss(seq_length, prediction_window)
    @test loss(nn, input_mat, output_mat) ≈
          norm(output_mat - nn(input_mat)[:, (seq_length - prediction_window + 1):end]) /
          norm(output_mat)

    Random.seed!(123)
    N, n = 4, 1
    nn = NeuralNetwork(SymplecticAutoencoder(2N, 2n))
    input_vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    loss = AutoEncoderLoss()
    @test loss(nn, input_vec) ≈ norm(input_vec - nn(input_vec)) / norm(input_vec)

    Random.seed!(123)
    Ψᵉ = NeuralNetwork(Chain(Dense(N, n), Dense(n, n))) |> encoder
    Ψᵈ = NeuralNetwork(Chain(Dense(n, n), Dense(n, N))) |> decoder
    transformer = NeuralNetwork(StandardTransformerIntegrator(n))
    input_mat = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
    output_mat = [9.0 10.0; 11.0 12.0; 13.0 14.0; 15.0 16.0]
    loss = ReducedLoss(Ψᵉ, Ψᵈ)
    output_prediction = Ψᵈ(transformer(Ψᵉ(input_mat)))
    @test loss(transformer, input_mat, output_mat) ≈
          norm(output_mat - output_prediction) / norm(output_mat)

    Random.seed!(123)
    H(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ X[2, :]
    dec = decoder(NeuralNetwork(SymplecticAutoencoder(
        4, 2; n_encoder_blocks = 2, n_decoder_blocks = 2)))
    z = randn(2, 20)
    x = dec(z)
    g = reduce(hcat,
        [Zygote.gradient(ζ -> H(reshape(dec(ζ), 4, 1))[1], collect(ζ))[1]
         for ζ in eachcol(z)])
    loss = ReducedHamiltonianLoss(H)
    @test loss(dec, z, vcat(x, g)) < 1e-6

    Random.seed!(123)
    H₁(X) = vec(sum(abs2, X[3:4, :]; dims = 1)) ./ 2 .+ cos.(X[1, :])
    XH₁(X) = vcat(X[3:4, :], sin.(X[1:1, :]), zero(X[2:2, :]))
    arch = PSDArch(4, 2)
    nn = NeuralNetwork(arch)
    solve!(nn, randn(4, 20))
    x = decoder(nn)(randn(2, 10))
    loss = LatentVectorFieldLoss(arch, H₁)
    @test loss(nn, x, XH₁(x)) < 1e-8
end
