using Test
using GeometricMachineLearning
using GeometricMachineLearning: AutoEncoderLoss, QPT, ZygotePullback
import Random

Random.seed!(1234)

@testset "Utility and pullback docstring examples" begin
    data1 = (q = rand(5), p = rand(5))
    data2 = (q = rand(5, 4), p = rand(5, 4))
    data3 = (q = rand(5, 4, 2), p = rand(5, 4, 2))
    @test (typeof(data1) <: QPT, typeof(data2) <: QPT, typeof(data3) <: QPT) ==
          (true, true, true)

    𝕁 = PoissonTensor(4)
    @test 𝕁 * (q = [1, 2], p = [3, 4]) == (q = [3, 4], p = [-1, -2])

    # the `ZygotePullback` docstring: `back(Δ)` is a `NetworkParameters`, not a one-tuple of one
    pb = ZygotePullback(AutoEncoderLoss())
    nn = NeuralNetwork(Chain(Dense(10, 2, tanh), Dense(2, 10, tanh)))
    _, back = pb(nn.params, nn.model, rand(10))
    g = back(1)
    @test (g isa NetworkParameters, keys(g)) == (true, (:L1, :L2))
end
