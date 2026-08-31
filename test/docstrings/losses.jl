using Test
using GeometricMachineLearning
using LinearAlgebra: norm
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
end
