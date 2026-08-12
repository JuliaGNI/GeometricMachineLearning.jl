using Test
using GeometricMachineLearning: QPT, _processing

@testset "Utility and pullback docstring examples" begin
    data1 = (q = rand(5), p = rand(5))
    data2 = (q = rand(5, 4), p = rand(5, 4))
    data3 = (q = rand(5, 4, 2), p = rand(5, 4, 2))
    @test (typeof(data1) <: QPT, typeof(data2) <: QPT, typeof(data3) <: QPT) ==
        (true, true, true)

    𝕁 = PoissonTensor(4)
    @test 𝕁 * (q = [1, 2], p = [3, 4]) == (q = [3, 4], p = [-1, -2])

    loss = AutoEncoderLoss()
    _pullback = ZygotePullback(loss)
    nn = NeuralNetwork(Chain(Dense(10, 2, tanh), Dense(2, 10, tanh)))
    input = rand(10)
    processed = _pullback(nn.params, nn.model, input)[2](1) |> _processing
    @test processed isa NamedTuple
    @test keys(processed) == (:L1, :L2)
end
