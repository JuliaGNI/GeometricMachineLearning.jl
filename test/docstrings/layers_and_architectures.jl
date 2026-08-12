using Test
using GeometricMachineLearning: UnknownEncoder, params

@testset "Layer and architecture docstring examples" begin
    l = LinearSymplecticAttentionQ(3, 5)
    ps = params(NeuralNetwork(Chain(l))).L1
    @test ps.A isa SymmetricMatrix

    model = Chain(Dense(5, 3, tanh; use_bias = false), Dense(3, 2, identity; use_bias = false))
    nn = NeuralNetwork(UnknownEncoder(5, 2, 2), model, params(NeuralNetwork(model)), CPU())
    @test nn isa NeuralNetwork{<:GeometricMachineLearning.Encoder}

    model = ResNet(3, 0, identity)
    weight = [1 0 0; 0 2 0; 0 0 1]
    bias = [0, 0, 1]
    ps = NeuralNetworkParameters((L1 = (weight = weight, bias = bias),))
    nn = NeuralNetwork(model, Chain(model), ps, CPU())
    @test iterate(nn, [1, 1, 1]; n_points = 4) == [1 2 4 8; 1 3 9 27; 1 3 7 15]

    dl = DataLoader(rand(2, 20); suppress_info = true)
    @test LASympNet(dl) isa LASympNet
end
