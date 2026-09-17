using GeometricMachineLearning
using Test
using Zygote
import Random
Random.seed!(1234)

const GMLA = GeometricMachineLearning.Activation
function allocate_network_and_data_loader(dim::Integer, width::Integer, nhidden::Integer, activation::GMLA)
    arch = StandardHamiltonianArchitecture(dim, width, nhidden, activation)
    loss = HNNLoss(arch)
    input = rand(dim, 10)
    output = rand(dim, 10)
    dl = DataLoader(input, output)
    NeuralNetwork(arch), loss, dl
end

"""
This tests if we can call the HNN loss.
"""
function test_hnn_loss(dim::Integer = 2,
        width::Integer = dim,
        nhidden::Integer = 1,
        activation::GMLA = GeometricMachineLearning.SigmoidActivation())
    nn, loss, dl = allocate_network_and_data_loader(dim, width, nhidden, activation)
    @test typeof(loss(nn.params, dl.input, dl.output)) <: Real
    @test typeof(loss) <: NetworkLoss
end

test_hnn_loss()

"""
This tests if we can differentiate the HNN loss.
"""
function test_hnn_loss_derivative(dim::Integer = 2,
        width::Integer = dim,
        nhidden::Integer = 1,
        activation::GMLA = GeometricMachineLearning.SigmoidActivation())
    nn, loss, dl = allocate_network_and_data_loader(dim, width, nhidden, activation)
    dp = Zygote.gradient(ps -> loss(ps, dl.input, dl.output), nn.params)[1]
    @test typeof(dp) <: NetworkParameters
    @test keys(dp) == keys(nn.params)
end

test_hnn_loss_derivative()

"""
This tests if we can build the symbolic pullback of the HNN loss and evaluate it.
"""
function test_hnn_symbolic_pullback(dim::Integer = 2,
        width::Integer = dim,
        nhidden::Integer = 1,
        activation::GMLA = GeometricMachineLearning.SigmoidActivation())
    arch = StandardHamiltonianArchitecture(dim, width, nhidden, activation)
    nn = NeuralNetwork(arch)
    pb = SymbolicPullback(arch)
    input = rand(dim, 10)
    output = rand(dim, 10)
    loss_value, back = pb(nn.params, nn.model, (input, output))
    @test typeof(loss_value) <: Real
    @test keys(back(1)) == keys(nn.params)
end

test_hnn_symbolic_pullback()
