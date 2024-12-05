using GeometricMachineLearning
using Test
import Random 
Random.seed!(1234)

function test_hnn_loss( dim::Integer = 2, 
                        width::Integer = dim, 
                        nhidden::Integer = 1, 
                        activation::GeometricMachineLearning.Activation = GeometricMachineLearning.SigmoidActivation())
    arch = StandardHamiltonianArchitecture(dim, width, nhidden, activation)
    loss = HNNLoss(arch)
    ps = NeuralNetwork(arch).params
    input = rand(dim, 10)
    output = rand(dim, 10)
    @test typeof(loss(ps), input, output) <: Real
end

test_hnn_loss()